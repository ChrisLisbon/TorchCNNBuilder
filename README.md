# TorchCNNBuilder
<p align="center">

<img src=".docs/media/logo_transparent_h.PNG" width="600">
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
### Description in Russian is presented [here](README_RU.md).

---

**TorchCNNBuilder** is an open-source framework for the automatic creation of CNN architectures. 
This framework should first of all help researchers in the applicability of CNN models for a 
huge range of tasks, taking over most of the writing of the architecture code. 
This framework is distributed under the **3-Clause BSD license**. All the functionality is written 
only using `pytorch` *(no third-party dependencies)*.

### Installation

---
The simplest way to install framework is using `pip`:
```
pip install torchcnnbuilder
```

<details><summary>Минимальные системные требования</summary>
Минимальными системными требованиями для использования библиотеки является
наличие интерпретатора Python версии >3.9 и доступ к вычислительной системе 
под управлением OC Windows/Linux. Минимальные требования к аппаратному обеспечению 
включают наличие процессора (CPU) с 8 ядрами, оперативной памяти (RAM) 2ГБ, 
графического процессора (GPU) с 8 ГБ VRAM и хранилища HDD 2 ГБ.
</details>

<details><summary>Additional packages for examples run</summary>

Please note that when running examples from the [examples](examples) folder, 
additional libraries are used to visualize and generate the dataset:

```
pip install numpy
pip install pytorch_msssim
pip install matplotlib
pip install tqdm
```

They are not required for the library to work, so their installation is optional.

</details>

### Usage

To initialize simple model with encoder-decoder architecture call ```ForecasterBase``` class:
```python
from torchcnnbuilder.models import ForecasterBase

model = ForecasterBase(input_size=[H, W],
                       in_time_points=C_in,
                       out_time_points=C_out,
                       n_layers=5)
```
Where ```[H, W]``` - size of image in pixels, ```C_in``` - number of input channels, ```C_out``` - number of out_channels. 

To operate separately with encoder and decoder parts they can be called from model:
```python
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

### Application Areas

TorchCNNBuilder enables CNN architectures for diverse real-world applications across multiple domains:

#### Environmental Monitoring

- **Sea ice concentration forecasting**  
  Predict Arctic and Antarctic ice melt patterns to support climate research and maritime navigation safety using satellite imagery time series.

- **Climate pattern recognition**  
  Analyze large-scale atmospheric data to identify emerging weather patterns, extreme event precursors, and long-term climate trends.

- **Pollution level prediction**  
  Process multispectral sensor data to forecast air/water quality indices and identify pollution sources with spatial CNN architectures.

#### Remote Sensing

- **Satellite image analysis**  
  Process high-resolution multispectral imagery for applications ranging from urban planning to precision agriculture using specialized encoder architectures.

- **Land cover classification**  
  Automate large-scale terrain mapping with attention-based CNNs that handle spectral, spatial and temporal dimensions of data.

- **Disaster monitoring**  
  Develop change detection systems that compare pre/post-event satellite imagery to assess flood, fire or earthquake damage in near-real-time.


#### Medical Imaging

- **Automated diagnosis from X-ray/MRI scans**  
  Develop assistive diagnostic systems that can detect abnormalities in medical images with pixel-level precision while reducing radiologist workload.

- **Tumor segmentation**  
  Create 3D convolutional networks for precise volumetric analysis of cancerous growths in CT/MRI scans.

- **Medical time-series analysis**  
  Process sensor streams to predict patient deterioration through temporal features processing architectures.

#### Industrial Applications

- **Predictive maintenance**  
  Monitor equipment vibration patterns and thermal signatures to forecast mechanical failures.

- **Quality control in manufacturing**  
  Implement real-time visual inspection systems that detect defects in production lines.


#### Financial Forecasting

- **Time-series prediction**  
  Build hybrid CNN-LSTM architectures that extract both spatial patterns from market heatmaps and temporal dependencies from price histories.

- **Market trend analysis**  
  Process alternative data sources like satellite images of parking lots or social media sentiment through CNN architectures.

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