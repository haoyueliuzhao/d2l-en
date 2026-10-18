#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
uv python install 3.12.13
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12.13 .venv
fi
constraints=()
if [[ -f deployment/constraints.txt ]]; then
  constraints=(-c deployment/constraints.txt)
fi
uv pip install --python .venv/bin/python "${constraints[@]}" \
  --index-url https://download.pytorch.org/whl/cu128 \
  -r deployment/requirements-torch.in
uv pip install --python .venv/bin/python "${constraints[@]}" -e .
.venv/bin/python -m ipykernel install --sys-prefix \
  --name d2l-pytorch --display-name 'D2L PyTorch (CUDA 12.8)'
echo 'Learning environment installed. Build the book/notebooks next; see deployment/README.zh-CN.md.'
