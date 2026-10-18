from setuptools import setup, find_packages
import d2l

# Runtime dependencies for the modern, isolated learning deployment.
# Exact versions of the deployed environment are recorded under deployment/.
requirements = [
    'jupyterlab>=4.4,<5',
    'ipykernel>=6.29,<8',
    'numpy>=2.1,<3',
    'matplotlib>=3.9,<4',
    'matplotlib-inline>=0.1.7,<1',
    'requests>=2.32,<3',
    'pandas>=2.2,<4',
    'scipy>=1.14,<2'
]


setup(
    name='d2l',
    version=d2l.__version__,
    python_requires='>=3.12',
    author='D2L Developers',
    author_email='d2l.devs@gmail.com',
    url='https://d2l.ai',
    description='Dive into Deep Learning',
    license='MIT-0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
