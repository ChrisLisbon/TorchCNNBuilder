from torchcnnbuilder.version import __version__
from setuptools import setup, find_packages
from typing import List


def readme() -> str:
    with open('README.md', 'r') as f:
        return f.read()


def _get_requirements(file_name: str) -> List[str]:
    with open(file_name, 'r') as f:
        file = f.readlines()
    return [line for line in file if line and not line.startswith('#')]


NAME = 'torchcnnbuilder'
VERSION = __version__
AUTHOR = 'NSS Lab'
URL = 'https://github.com/ITMO-NSS-team/TorchCNNBuilder'
PYTHON_REQUIRES = '>=3.8'
REQUIREMENTS_PATH = 'requirements.txt'

setup(
    name=NAME,
    version=VERSION,
    author='NSS Lab',
    author_email='just.andrew.kd@gmail.com',
    description='Framework for the automatic creation of CNN architectures',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url=URL,
    packages=find_packages(),
    install_requires=_get_requirements(REQUIREMENTS_PATH),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    keywords='python torch cnn',
    python_requires=PYTHON_REQUIRES
)
