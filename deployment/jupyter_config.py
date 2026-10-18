"""Private, loopback-only JupyterLab configuration used by server.py."""

import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / ".d2l-runtime"
TOKEN_FILE = Path(os.environ.get("D2L_JUPYTER_TOKEN_FILE", RUNTIME / "jupyter.token"))

if not TOKEN_FILE.is_file():
    raise RuntimeError("Start JupyterLab with: python deployment/server.py start jupyter")
token = TOKEN_FILE.read_text(encoding="utf-8").strip()
if not token:
    raise RuntimeError("The Jupyter authentication token must not be empty.")

c = get_config()  # noqa: F821 - supplied by Jupyter's config loader
c.ServerApp.ip = "127.0.0.1"
c.ServerApp.port = 8888
c.ServerApp.port_retries = 0
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.allow_remote_access = False
c.ServerApp.root_dir = str(REPO)
c.ServerApp.default_url = "/lab/tree/notebooks/pytorch"
c.ServerApp.runtime_dir = str(RUNTIME / "jupyter")
c.ServerApp.log_level = "WARN"
c.IdentityProvider.token = token
c.PasswordIdentityProvider.allow_password_change = False

# Reuse Jupyter's authenticated origin for the already-running textbook server.
c.ServerApp.jpserver_extensions = {"jupyter_server_proxy": True}
c.ServerProxy.servers = {
    "book": {
        "command": [],
        "port": int(os.environ.get("D2L_BOOK_PORT", "8080")),
        "absolute_url": False,
        "launcher_entry": {"title": "D2L Textbook"},
        "new_browser_tab": True,
    }
}
