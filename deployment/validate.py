#!/usr/bin/env python3
"""Bounded deployment acceptance: representative notebooks and real CUDA work."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('MPLBACKEND', 'Agg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebooks', type=Path, default=ROOT / 'notebooks/pytorch')
    args = parser.parse_args()
    import nbformat
    from nbclient import NotebookClient
    import numpy as np
    import pandas as pd
    import torch
    import torchvision
    from d2l import torch as d2l
    report = {
        'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'cuda_runtime': torch.version.cuda,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'd2l_module': str(Path(d2l.__file__).relative_to(ROOT)),
        'scope': 'Two CPU textbook notebooks and a small synthetic CUDA regression; not a full-book or performance benchmark.',
    }
    assert torch.cuda.is_available(), 'CUDA unavailable'
    torch.set_num_threads(8)
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    report['gpu'] = torch.cuda.get_device_name(0)
    report['compute_capability'] = list(torch.cuda.get_device_capability(0))
    report['compiled_cuda_architectures'] = torch.cuda.get_arch_list()
    # Exercise matrix multiplication, autograd and optimizer kernels on the GPU.
    x = torch.randn(1024, 2, device=device)
    expected_w = torch.tensor([[2.0], [-3.4]], device=device)
    y = x @ expected_w + 4.2
    model = torch.nn.Linear(2, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial = torch.nn.functional.mse_loss(model(x), y).item()
    start = time.monotonic()
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    final = torch.nn.functional.mse_loss(model(x), y).item()
    assert final < 1e-6 and final < initial * 1e-4, (initial, final)
    # Validate a compiled torchvision CUDA operator, not just a torch import.
    kept = torchvision.ops.nms(
        torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=device),
        torch.tensor([0.9, 0.8], device=device), 0.5)
    assert kept.tolist() == [0]
    report['gpu_regression'] = {
        'examples': 1024, 'steps': 100, 'seed': 42,
        'initial_mse': initial, 'final_mse': final,
        'elapsed_seconds': round(time.monotonic() - start, 3),
        'device': str(device), 'torchvision_cuda_nms': 'passed',
    }
    features, labels = d2l.synthetic_data(torch.tensor([2., -3.4]), 4.2, 10)
    assert features.shape == (10, 2) and labels.shape == (10, 1)
    report['d2l_synthetic_data'] = 'passed'
    output = ROOT / '.deployment-validation'
    output.mkdir(exist_ok=True)
    report['notebooks'] = []
    for name in ('chapter_preliminaries/ndarray.ipynb', 'chapter_preliminaries/autograd.ipynb'):
        path = args.notebooks / name
        notebook = nbformat.read(path, as_version=4)
        start = time.monotonic()
        NotebookClient(notebook, timeout=180, kernel_name='d2l-pytorch',
                       resources={'metadata': {'path': str(path.parent)}}).execute()
        nbformat.write(notebook, output / path.name)
        report['notebooks'].append({
            'path': name, 'status': 'passed',
            'code_cells': sum(c.cell_type == 'code' for c in notebook.cells),
            'elapsed_seconds': round(time.monotonic() - start, 3),
        })
    report['status'] = 'passed'
    (ROOT / 'deployment/validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
