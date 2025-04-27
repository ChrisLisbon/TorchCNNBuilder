**TorchCNNBuilder** is an open-source framework for the automatic creation of CNN architectures.
This framework should first of all help researchers in the applicability of CNN models for a huge range of tasks,
taking over most of the writing of the architecture code. This framework is distributed under the 3-Clause BSD license.
All the functionality is written only using `pytorch` *(no third-party dependencies)*.

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

You can check current package version by using constant `__version__`:
```python
from torchcnnbuilder import __version__

print(__version__)
```

# Development

We try to maintain good practices of readable open source code. 
Therefore, if you want to participate in the development and open your pool request, pay attention to the following points:
- Every push is checked by the flake8 job. It will show you PEP8 errors or possible code improvements.
- Use this linter script in the repo root after your code *(it needs some extra dependencies)*:

```bash
make lint
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

Our doc is created by using `pdoc` framework. In order to **build** and **serve** 
the doc locally run in the project root:
```sh
make doc
```
This command is checking and installing `pdoc`, then is building and serving the doc.
