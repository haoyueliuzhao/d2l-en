#!/usr/bin/env python3
"""Manage the local D2L textbook and JupyterLab without systemd.

Examples:
    python deployment/server.py start
    python deployment/server.py status
    python deployment/server.py url jupyter
    python deployment/server.py stop

Services bind only to 127.0.0.1. Use SSH port forwarding for remote access.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / ".d2l-runtime"
PYTHON = REPO / ".venv" / "bin" / "python"
HOST = "127.0.0.1"
TOKEN_FILE = RUNTIME / "jupyter.token"
SERVICES = ("jupyter", "book")


class ServiceError(Exception):
    """An actionable service management error."""


def private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ServiceError(f"Refusing a symlink for private runtime directory: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def private_open(path: Path, flags: int) -> int:
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)
    os.fchmod(fd, 0o600)
    return fd


def private_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with os.fdopen(private_open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "w") as handle:
            handle.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def management_lock():
    private_directory(RUNTIME)
    with os.fdopen(private_open(RUNTIME / "manage.lock", os.O_WRONLY | os.O_CREAT), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def state_path(name: str) -> Path:
    return RUNTIME / f"{name}.json"


def read_state(name: str) -> dict | None:
    path = state_path(name)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(state, dict)
            or state.get("name") != name
            or type(state.get("pid")) is not int
            or state["pid"] <= 1
            or not isinstance(state.get("start_time"), str)
            or not isinstance(state.get("command"), list)
            or not state["command"]
            or not all(isinstance(arg, str) for arg in state["command"])
            or type(state.get("port")) is not int
            or not 1 <= state["port"] <= 65535
        ):
            raise ValueError("invalid fields")
        return state
    except (ValueError, OSError) as error:
        raise ServiceError(f"Cannot read {path}: {error}. Inspect this state file before removing it.") from error


def process_snapshot(pid: int) -> tuple[str, list[str]] | None:
    """Read Linux process identity, including start ticks to detect PID reuse."""
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        command = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return fields[19], [os.fsdecode(arg) for arg in command]
    except (OSError, IndexError):
        return None


def is_running(state: dict | None) -> bool:
    if state is None:
        return False
    return process_snapshot(state["pid"]) == (state["start_time"], state["command"])


def base_url(state: dict) -> str:
    return f"http://{HOST}:{state['port']}"


def read_token() -> str:
    try:
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ServiceError(f"Cannot read Jupyter token: {error}") from error
    if not token:
        raise ServiceError("Jupyter token is empty; stop and start the service to recreate it.")
    return token


def ready(state: dict) -> bool:
    if not is_running(state):
        return False
    url = base_url(state) + ("/api/status" if state["name"] == "jupyter" else "/")
    request = urllib.request.Request(url)
    if state["name"] == "jupyter":
        request.add_header("Authorization", f"token {read_token()}")
    try:
        # Never route loopback health checks through a configured HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=1) as response:
            return response.status == 200 and is_running(state)
    except (urllib.error.URLError, OSError):
        return False


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Match the HTTP server: recently closed connections may remain in TIME_WAIT.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((HOST, port))
        except OSError:
            return False
    return True


def terminate(state: dict) -> None:
    if not is_running(state):
        return
    pid = state["pid"]
    # We only create session leaders. Never signal an unrelated process group.
    try:
        if os.getpgid(pid) != pid or os.getsid(pid) != pid:
            raise ServiceError(f"Refusing to stop PID {pid}: its session identity changed.")
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while is_running(state) and time.monotonic() < deadline:
        time.sleep(0.1)
    if is_running(state):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 3
        while is_running(state) and time.monotonic() < deadline:
            time.sleep(0.1)
        if is_running(state):
            raise ServiceError(f"PID {pid} did not exit; retaining its state file.")


def start(name: str, args: argparse.Namespace, *, preserve_token: bool = False) -> None:
    previous = read_state(name)
    if is_running(previous):
        if not ready(previous):
            raise ServiceError(f"{name}: recorded process is running but is not ready; see {RUNTIME / (name + '.log')}")
        print(f"{name}: already running (PID {previous['pid']}) at {base_url(previous)}")
        return
    if previous:
        print(f"{name}: removing stale process record; no process was signalled")
        state_path(name).unlink(missing_ok=True)
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise ServiceError(f"Python environment missing: {PYTHON}. Complete deployment setup first.")
    port = args.jupyter_port if name == "jupyter" else args.book_port
    if not port_available(port):
        raise ServiceError(f"{name}: {HOST}:{port} is already occupied; no existing process was changed.")

    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    if name == "jupyter":
        private_directory(RUNTIME / "jupyter")
        if not preserve_token or not TOKEN_FILE.exists():
            private_write(TOKEN_FILE, secrets.token_urlsafe(32) + "\n")
        else:
            read_token()  # A restart must not silently disable authentication.
        book_state = read_state("book")
        book_port = book_state["port"] if is_running(book_state) else args.book_port
        environment["D2L_BOOK_PORT"] = str(book_port)
        environment["D2L_JUPYTER_TOKEN_FILE"] = str(TOKEN_FILE)
        environment["JUPYTER_RUNTIME_DIR"] = str(RUNTIME / "jupyter")
        command = [
            str(PYTHON), "-m", "jupyterlab", "--config",
            str(REPO / "deployment" / "jupyter_config.py"), f"--ServerApp.port={port}",
        ]
    else:
        book_dir = Path(args.book_dir).expanduser()
        if not book_dir.is_absolute():
            book_dir = REPO / book_dir
        book_dir = book_dir.resolve()
        if book_dir == REPO or REPO.is_relative_to(book_dir):
            raise ServiceError("The textbook directory must not expose the repository root or its ancestors.")
        if not (book_dir / "index.html").is_file():
            raise ServiceError(f"Textbook index missing: {book_dir / 'index.html'}. Build the textbook or set --book-dir.")
        command = [str(PYTHON), "-m", "http.server", str(port), "--bind", HOST, "--directory", str(book_dir)]

    log_path = RUNTIME / f"{name}.log"
    with os.fdopen(private_open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND), "ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_snapshot(process.pid)
    if identity is None:
        process.wait(timeout=5)
        raise ServiceError(f"{name}: process exited immediately; see {log_path}")
    state = {"name": name, "pid": process.pid, "start_time": identity[0], "command": command, "port": port}
    private_write(state_path(name), json.dumps(state, indent=2) + "\n")
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ServiceError(f"{name}: process exited with code {process.returncode}; see {log_path}")
            if ready(state):
                print(f"{name}: started (PID {process.pid}) at {base_url(state)}")
                return
            time.sleep(0.25)
        raise ServiceError(f"{name}: not ready after {args.timeout:g} seconds; see {log_path}")
    except BaseException:
        terminate(state)
        process.wait(timeout=5)
        state_path(name).unlink(missing_ok=True)
        if name == "jupyter":
            TOKEN_FILE.unlink(missing_ok=True)
        raise


def stop(name: str, *, preserve_token: bool = False) -> None:
    state = read_state(name)
    if is_running(state):
        terminate(state)
        print(f"{name}: stopped")
    else:
        print(f"{name}: not running")
    state_path(name).unlink(missing_ok=True)
    if name == "jupyter" and not preserve_token:
        TOKEN_FILE.unlink(missing_ok=True)


def status(name: str) -> bool:
    state = read_state(name)
    if not is_running(state):
        print(f"{name}: stopped" + (" (stale process record)" if state else ""))
        return False
    healthy = ready(state)
    print(f"{name}: {'running' if healthy else 'not ready'} (PID {state['pid']}) at {base_url(state)}")
    return healthy


def show_url(name: str) -> None:
    state = read_state(name)
    if not is_running(state):
        raise ServiceError(f"{name}: not running")
    url = base_url(state)
    if name == "jupyter":
        url += "/lab/tree/notebooks/pytorch?" + urllib.parse.urlencode({"token": read_token()})
    elif name == "book":
        jupyter = read_state("jupyter")
        if is_running(jupyter):
            url = base_url(jupyter) + "/book/?" + urllib.parse.urlencode({"token": read_token()})
    print(f"{name}: {url}")


def port_number(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("start", "restart", "status", "stop", "url"))
    parser.add_argument("service", nargs="?", default="all", choices=("all", *SERVICES))
    parser.add_argument("--book-dir", default="_build/html", help="textbook directory, relative to repository (default: _build/html)")
    parser.add_argument("--jupyter-port", type=port_number, default=8888)
    parser.add_argument("--book-port", type=port_number, default=8080)
    parser.add_argument("--timeout", type=float, default=60, help="startup timeout per service in seconds (default: 60)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    names = SERVICES if args.service == "all" else (args.service,)
    success = True
    try:
        with management_lock():
            for name in names:
                try:
                    if args.action == "start":
                        start(name, args)
                    elif args.action == "restart":
                        previous = read_state(name)
                        if is_running(previous):
                            if name == "jupyter":
                                args.jupyter_port = previous["port"]
                            else:
                                args.book_port = previous["port"]
                                args.book_dir = previous["command"][-1]
                        stop(name, preserve_token=True)
                        start(name, args, preserve_token=True)
                    elif args.action == "stop":
                        stop(name)
                    elif args.action == "status":
                        success = status(name) and success
                    else:
                        show_url(name)
                except (ServiceError, OSError, subprocess.SubprocessError) as error:
                    print(f"Error: {error}", file=sys.stderr)
                    success = False
        if args.action in ("start", "restart") and "jupyter" in names and success:
            print("For the authenticated Jupyter URL: python deployment/server.py url jupyter")
    except (ServiceError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
