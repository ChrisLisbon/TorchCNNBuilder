# Installation

The simplest way to install framework is using `pip`:
```
pip install torchcnnbuilder
```

# Purposes 

Initially, the library was created to help predict n-dimensional time series *(geodata)*, so there is a corresponding functionality and templates of predictive models *(like `ForecasterBase`)*.
Basic framework functions are presented below: 

- the ability to calculate the size of tensors after (transposed) convolutional layers
- preprocessing an n-dimensional time series in `TensorDataset` (`torchcnnbuilder.preprocess`)
- automatic creation of (transposed) convolutional sequences (`torchcnnbuilder.builder`)
- automatic creation of (transposed) convolutional layers and (transposed) blocks from convolutional layers (`torchcnnbuilder.preprocess`)
- automatic creation of convolution encoder-decoder models (`torchcnnbuilder.models`)
- the ability to change latent space params after/before encoder/decoder parts (`torchcnnbuilder.latent`)

# Constants 

You can check current package version by using constant `__version__`:
```python
from torchcnnbuilder import __version__

print(__version__)
# output: 0.1.2
```

Also you can check default torch convolution/transpose convolution params:
```python
from torchcnnbuilder import DEFAULT_CONV_PARAMS, DEFAULT_TRANSPOSE_CONV_PARAMS

print(DEFAULT_CONV_PARAMS)
# output: {'kernel_size': 3, 'stride': 1, 'padding': 0, 'dilation': 1}

print(DEFAULT_TRANSPOSE_CONV_PARAMS)
# output: {'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0, 'dilation': 1}
```

# Development

We try to maintain good practices of readable open source code. 
Therefore, if you want to participate in the development and open your pool request, pay attention to the following points:
- Every push is checked by the flake8 job. It will show you PEP8 errors or possible code improvements.
- Use this linter script in the repo root after your code:

```bash
bash lint_and_check.sh
```
*You can mark function docstrings using `#noqa`, in order for flake8 not to pay attention to them.*

## General tips

- If it's possible, try to create pull-requests by using fork
- Give only appropriate names to commits / issues / pull-requests
- It's better to use `pyenv`, `conda` or some different options of python environments in order to develop

## Release process

Despite the fact that the framework is very small, we want to maintain its consistency. 
The release procedure looks like this:

- pull-request is approved by maintainers and merged with squashing commits
- a new tag is being released to the github repository
- a new tag is being released in pypi

## Building doc 

Our doc is created by using `pdoc` framework. In order to **build** and **serve** the doc locally:
- Install `pdoc`:
```bash
pip install pdoc==15.0.0
```
- Run in the repo root:
```bash
pdoc --math -d google --no-include-undocumented -t .docs/ ./torchcnnbuilder
```
