
# TorchCNNBuilder
<p align="center">

<img src=".docs/media/logo_transparent.PNG" width="300">
</p>

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
**TorchCNNBuilder** is an open-source framework for the automatic creation of CNN architectures. This framework should first of all help researchers in the applicability of CNN models for a huge range of tasks, taking over most of the writing of the architecture code. This framework is distributed under the 3-Clause BSD license. All the functionality is written only using `pytorch` *(no third-party dependencies)*.

### Installation

---
The simplest way to install framework is using `pip`:
```
pip install torchcnnbuilder
```


### Usage

To initialize simple model with encoder-decoder architecture call ```ForecasterBase``` class:
```
model = ForecasterBase(input_size=[H, W],
                       in_time_points=C_in,
                       out_time_points=C_out,
                       n_layers=5)
```
Where ```[H, W]``` - size of image in pixels, ```C_in``` - number of input channels, ```C_out``` - number of out_channels. 

To operate separately with encoder and decoder parts they can be called from model:
```
encoder = model.encoder
decoder = model.decoder
```

### Examples

Extensive usage scenarios can be found in [examples](examples) folder.

Components calls and usage in folder [usage_examples](examples/usage_examples).


### Documentation 

Check the documentation [here](https://chrislisbon.github.io/TorchCNNBuilder/torchcnnbuilder.html). 

### Development 

In order to check available local `Makefile` commands run in the project root: 
```sh
make help
```
```yaml
help: Show help for each of the Makefile recipes.
lint: Lint the project with flake8 lib.
doc: Build and run the doc locally.
```

### Sources

---
- [Forecasting of Sea Ice Concentration using CNN, PDE discovery and Bayesian Networks](https://www.sciencedirect.com/science/article/pii/S1877050923020094)
- [Surrogate Modelling for Sea Ice Concentration using Lightweight Neural Ensemble](https://arxiv.org/abs/2312.04330)
- [Post about framework development on habr.com - in russian](https://habr.com/ru/companies/selectel/articles/818649/)


### Contributing

- To join the project feel free to [contact us](mailto:jul.borisova@itmo.ru);

- [Issues](https://github.com/ChrisLisbon/TorchCNNBuilder/issues) and 
[Pull Requests](https://github.com/ChrisLisbon/TorchCNNBuilder/pulls): submit bugs found or log feature requests.

### Acknowledgement

The project is supported by [FASIE](https://fasie.ru/) - Foundation for Assistance to Small Innovative Enterprises.