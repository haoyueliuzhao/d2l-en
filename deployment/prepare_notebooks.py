#!/usr/bin/env python3
"""Create editable notebook copies, preserving every existing user file."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '_build/notebooks/pytorch'
DESTINATION = ROOT / 'notebooks/pytorch'


def main():
    if not (SOURCE / 'index.ipynb').is_file():
        raise SystemExit('Build the book first: .venv-book/bin/python deployment/build_book.py')
    copied = preserved = 0
    for source in sorted(SOURCE.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts:
            continue
        target = DESTINATION / source.relative_to(SOURCE)
        if target.exists():
            preserved += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.ipynb':
            notebook = json.loads(source.read_text())
            notebook.setdefault('metadata', {})['kernelspec'] = {
                'display_name': 'D2L PyTorch (CUDA 12.8)',
                'language': 'python', 'name': 'd2l-pytorch',
            }
            target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
        else:
            shutil.copy2(source, target)
        copied += 1
    print(f'Copied {copied} files; preserved {preserved} existing files in {DESTINATION}')


if __name__ == '__main__':
    main()
