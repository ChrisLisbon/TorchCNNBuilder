# TorchCNNBuilder

<div id="badges">
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/pytorch-CB2C31?style=flat&logo=pytorch&logoColor=white" alt="pytorch badge"/>
    </a>
    <img alt="Dynamic JSON Badge" src="https://img.shields.io/pypi/pyversions/torch">
    <a href="https://badge.fury.io/py/torchcnnbuilder">
        <img src="https://badge.fury.io/py/torchcnnbuilder.svg" alt="PyPI version" height="18">
    </a>
</div>

---
**TorchCNNBuilder** is an open-source framework for the automatic creation of CNN architectures. This framework should first of all help researchers in the applicability of CNN models for a huge range of tasks, taking over most of the writing of the architecture code. This framework is distributed under the 3-Clause BSD license. All the functionality is written only using `pytorch` *(no third-party dependencies)*

### Installation

---
The simplest way to install framework is using `pip`:
```
pip install torchcnnbuilder
```

### Usage examples
---
The basic structure of the framework is presented below. Each subdirectory has its own example of using the appropriate available functionality. You can check `<directory>_examples.ipynb` files in order to see the ways to use the proposed toolkit. In short, there is the following functionality:

- the ability to calculate the size of tensors after (transposed) convolutional layers
- preprocessing an n-dimensional time series in `TensorDataset`
- automatic creation of (transposed) convolutional sequences
- automatic creation of (transposed) convolutional layers and (transposed) blocks from convolutional layers

The structure of the main part of the package:

```
├── examples
│ ├── builder_examples.ipynb
│ ├── preprocess_examples.ipynb
│ ├── models_examples.ipynb
│ └── tools                     # additional functions for the examples
└── torchcnnbuilder
    ├── preprocess
    │ └── time_series.py
    ├── builder.py
    └── models.py
```
Initially, the library was created to help predict n-dimensional time series *(geodata)*, so there is a corresponding functionality and templates of predictive models *(like `ForecasterBase`)*

### To Do
---
- [ ] add support for pulling and sampling layers
- [ ] add more templates for different CNN-models
- [ ] add more usage examples
- [ ] add auto tests

### Sources
---
- [Surrogate Modelling for Sea Ice Concentration using Lightweight Neural Ensemble](https://arxiv.org/abs/2312.04330)

