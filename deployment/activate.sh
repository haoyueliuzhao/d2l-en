# Source this file: source deployment/activate.sh
_d2l_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_d2l_root/.venv/bin/activate"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost,::1"
unset _d2l_root
