#!/usr/bin/env python3
"""Build all book tabs and runnable PyTorch notebooks without executing examples."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / '_build'
TABS = ('pytorch', 'mxnet', 'jax', 'tensorflow')
UPSTREAM_COMMIT = '23d7a5aecceee57d1292c56e90cce307f183bb0a'
UPSTREAM_URL = 'https://github.com/d2l-ai/d2l-en'


def configure(tab):
    from d2lbook.config import Config
    config = Config(tab=tab)
    config.build['eval_notebook'] = 'False'
    config.build['notebooks'] = 'index.md chapter_*/*.md'
    config.build['resources'] = 'img/ d2l/ d2l.bib'
    config.build['dependencies'] = 'deployment/build_book.py config.ini'
    config.pdf['style'] = ''  # HTML needs no Cambridge LaTeX download.
    config.colab['github_repo'] = ''
    config.sagemaker['github_repo'] = ''
    del config.deploy['google_analytics_tracking_id']
    config.html['header_links'] = (
        'Original book, https://d2l.ai, fas fa-book,\n'
        'PyTorch notebooks, /d2l-pytorch-notebooks.zip, fab fa-python,\n'
        f'Source, {UPSTREAM_URL}/tree/{UPSTREAM_COMMIT}, fab fa-github,\n'
        '中文版, https://zh.d2l.ai, fas fa-external-link-alt'
    )
    return config


def convert_tab(tab):
    """Use upstream tab selection and aliases without starting a kernel."""
    from d2lbook.build import Builder
    Builder(configure(tab)).eval()
    return tab


def convert_rst(pair):
    from d2lbook.build import ipynb2rst
    source, target = pair
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    ipynb2rst(str(source), str(target))
    # Pandoc 3 emits the D2L label as the table caption. Restore caption + target.
    content = Path(target).read_text()
    pattern = r'^:([^\n]+(?:\n[^\n]+)*)\n\n\.\. table:: label:``([^`]+)``'
    def restore_table(match):
        caption, label = match.groups()
        return f'.. _{label}:\n\n.. table:: ' + caption.replace('\n', '\n   ')
    Path(target).write_text(re.sub(pattern, restore_table, content, flags=re.M))
    return str(target)


def static_api_reference(destination):
    """Render source signatures/docstrings without importing training frameworks."""
    lines = [
        '.. _sec_d2l:', '', 'The ``d2l`` API Document',
        '========================', '',
        'This reference is extracted from the checked-out Python source. It lists',
        'defined classes, methods, and functions without importing or executing',
        'training frameworks; imported aliases and runtime-injected members may',
        'not be listed. See the source for complete implementations.', '',
    ]
    for module in ('torch', 'mxnet', 'jax', 'tensorflow'):
        filename = ROOT / 'd2l' / f'{module}.py'
        tree = ast.parse(filename.read_text())
        heading = f'd2l.{module}'
        lines.extend([heading, '-' * len(heading), '',
                      f'`Source for d2l.{module} <{UPSTREAM_URL}/blob/{UPSTREAM_COMMIT}/d2l/{module}.py>`_', ''])
        # Later definitions replace earlier definitions in these generated files.
        nodes = {node.name: node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                 and not node.name.startswith('_')}
        for name, node in sorted(nodes.items()):
            full_name = f'd2l.{module}.{name}'
            if isinstance(node, ast.ClassDef):
                lines.extend([f'.. py:class:: {full_name}', ''])
            else:
                lines.extend([f'.. py:function:: {full_name}({ast.unparse(node.args)})', ''])
            doc = ast.get_docstring(node)
            if doc:
                lines.extend('   ' + line for line in doc.splitlines())
                lines.append('')
            if isinstance(node, ast.ClassDef):
                methods = {method.name: method for method in node.body
                           if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))}
                for method in methods.values():
                    if method.name.startswith('_') and method.name != '__init__':
                        continue
                    signature = ast.unparse(method.args)
                    lines.extend([f'   .. py:method:: {method.name}({signature})', ''])
                    doc = ast.get_docstring(method)
                    if doc:
                        lines.extend('      ' + line for line in doc.splitlines())
                        lines.append('')
    destination.write_text('\n'.join(lines) + '\n')


def prepare_frontpage(destination):
    page = (ROOT / 'static/frontpage/frontpage.html').read_text()
    # Preserve the official layout and bundled static map; remove obsolete embeds.
    page = re.sub(r'<!--beginning of the map-->.*?<!--end of the map-->', '', page, flags=re.S)
    page = re.sub(r'<script\b[^>]*\bsrc=["\'][^"\']+["\'][^>]*>\s*</script>', '', page, flags=re.S)
    banner = ('<p style="padding: 1rem; background: #fff7e6; border-left: 4px solid #d48c16">'
              'Local reading edition: examples have not been executed during this build. '
              'Run the downloadable PyTorch notebooks to generate results and plots.</p>')
    destination.write_text(banner + '\n' + page)


def export_notebooks():
    from d2lbook.build import update_ipynb_toc
    import nbformat
    target = BUILD / 'notebooks/pytorch'
    target.mkdir(parents=True, exist_ok=True)
    source = BUILD / 'eval'
    current = set()
    for path in source.rglob('*.ipynb'):
        if not path.stat().st_size:
            continue  # Upstream uses empty files to mark unavailable framework tabs.
        relative = path.relative_to(source)
        current.add(relative)
        notebook = nbformat.read(path, as_version=4)
        notebook.metadata['kernelspec'] = {
            'display_name': 'D2L (PyTorch)', 'language': 'python', 'name': 'd2l-pytorch',
        }
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.execution_count = None
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
    for old in target.rglob('*.ipynb'):
        if old.relative_to(target) not in current:
            old.unlink()
    update_ipynb_toc(str(target))
    for resource in ('img', 'd2l'):
        shutil.copytree(ROOT / resource, target / resource, dirs_exist_ok=True)
    for filename in ('LICENSE', 'LICENSE-SAMPLECODE', 'LICENSE-SUMMARY'):
        shutil.copy2(ROOT / filename, target / filename)
    (target / 'README.txt').write_text(
        'Generated from the checked-out D2L source without executing examples.\n'
        'Use the D2L (PyTorch) kernel from the project learning environment.\n'
        'Some framework-specific chapters have no PyTorch implementation.\n'
        'Training chapters may download datasets or require GPU resources.\n'
    )
    return target, len(current)


def validate(html, notebook_count):
    required = ('index.html', 'search.html', 'searchindex.js',
                'chapter_preliminaries/ndarray.html',
                'chapter_preliminaries/autograd.html',
                'chapter_appendix-tools-for-deep-learning/d2l.html',
                '_images/front-cup.jpg', '_images/map.png')
    for relative in required:
        path = html / relative
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f'Missing build output: {relative}')
    js = (html / '_static/d2l.js').read_text()
    if 'google-analytics.com' in js or 'UA-96378503-10' in js:
        raise RuntimeError('Upstream analytics unexpectedly present')
    for page in html.rglob('*.html'):
        if '###_CURRENT_VERSION' in page.read_text():
            raise RuntimeError(f'Unexpanded navigation placeholder: {page}')
    if notebook_count == 0:
        raise RuntimeError(f'Unexpectedly few notebooks: {notebook_count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=min(16, os.cpu_count() or 2))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    os.chdir(ROOT)
    (BUILD / 'logs').mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(BUILD / 'logs/book-build.log', mode='w'), logging.StreamHandler()])
    import pypandoc
    pandoc_dir = str(Path(pypandoc.get_pandoc_path()).parent)
    os.environ['PATH'] = os.pathsep.join((str(Path(sys.executable).parent), pandoc_dir, os.environ['PATH']))
    from d2lbook.build import Builder
    from d2lbook.sphinx import prepare_sphinx_env
    from d2lbook import sphinx_template
    # Upstream tab handlers call these functions even when analytics is disabled.
    sphinx_template.google_tracker = ''
    sphinx_template.replace_qr = ''
    sphinx_template.discourse_js = ''
    logging.info('Converting all framework tabs with code execution disabled')
    with ProcessPoolExecutor(max_workers=min(4, args.jobs)) as pool:
        list(pool.map(convert_tab, TABS))
    notebooks, notebook_count = export_notebooks()
    logging.info('PyTorch notebooks ready: %s (%d)', notebooks, notebook_count)
    config = configure('all')
    builder = Builder(config)
    builder.merge()
    rst_root = Path(config.rst_dir)
    jobs = []
    for source in Path(config.eval_dir).rglob('*.ipynb'):
        if source.stat().st_size:
            target = rst_root / source.relative_to(config.eval_dir).with_suffix('.rst')
            if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
                jobs.append((source, target))
    logging.info('Converting %d pages with %d worker processes', len(jobs), args.jobs)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(convert_rst, jobs))
    prepare_sphinx_env(config)
    builder._copy_resources(config.src_dir, config.rst_dir)
    prepare_frontpage(rst_root / 'frontpage.html')
    static_api_reference(rst_root / 'chapter_appendix-tools-for-deep-learning/d2l.rst')
    with (rst_root / '_static/d2l.js').open('a') as stream:
        stream.write('\nfunction tagClick(tab) {}\nfunction captureOutboundLink(url) { window.location.href = url; }\n')
    html = Path(config.html_dir)
    sphinx_log = BUILD / 'logs/sphinx.log'
    with sphinx_log.open('w') as stream:
        subprocess.run([str(Path(sys.executable).parent / 'sphinx-build'), '-b', 'html',
                        '-j', str(args.jobs), str(rst_root), str(html)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    shutil.copytree(ROOT / 'static/frontpage/_images', html / '_images', dirs_exist_ok=True)
    # Resolve navbar download relative to each page, allowing hosting under a path prefix.
    for page in html.rglob('*.html'):
        content = page.read_text()
        relative_zip = os.path.relpath(html / 'd2l-pytorch-notebooks.zip', page.parent)
        page.write_text(content.replace('href="/d2l-pytorch-notebooks.zip"', f'href="{relative_zip}"'))
    (html / '.nojekyll').touch()
    with zipfile.ZipFile(html / 'd2l-pytorch-notebooks.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in notebooks.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:
                archive.write(file, Path('pytorch') / file.relative_to(notebooks))
    validate(html, notebook_count)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    report = {
        'source_commit': revision, 'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'html_directory': str(html.resolve()), 'html_pages': len(list(html.rglob('*.html'))),
        'pytorch_notebook_directory': str(notebooks), 'pytorch_notebooks': notebook_count,
        'notebook_execution': False, 'sphinx_log': str(sphinx_log),
        'framework_tabs': list(TABS), 'api_reference': 'AST source extraction without framework imports',
    }
    (BUILD / 'book-build-report.json').write_text(json.dumps(report, indent=2) + '\n')
    logging.info('Build complete: %s', report)


if __name__ == '__main__':
    main()
