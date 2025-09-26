[[Click For Russian version]](README_RU.md)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/ChrisLisbon/TorchCNNBuilder.git
   cd TorchCNNBuilder
   
2. Install in development mode (recommended):
    ```bash
   pip install -e .

**This will**: install all required dependencies, make the package available system-wide, allow 
you to modify code and see changes immediately.

3. For just running examples (without development):
    ```bash
    pip install numpy torch matplotlib jupyter
   
4. ⚠️ Note: The examples use relative paths for data loading. 
To avoid file not found errors, always launch Jupyter from the repository root directory:
    ```bash
    jupyter notebook examples/example_name.ipynb

# Usage examples

Relevant usage examples can be found in this directory. The API examples of each submodule are located in the corresponding `ipynb` file:
- [`torchcnnbuilder`](usage_examples/main_examples.ipynb) - main constants and convolution formulas
- [`torchcnnbuilder.builder`](usage_examples/builder_examples.ipynb) - builder API for the creation of convolution sequences
- [`torchcnnbuilder.models`](usage_examples/model_examples.ipynb) - examples of adaptive models created with builder API 
- [`torchcnnbuilder.preprocess`](usage_examples/preprocess_examples.ipynb) - data preprocessing

## Lightweight domain examples

Examples of models building and training:
- [`synthetic_noise_examples`](synthetic_noise_examples) - Architecture selection and noise resistance experiment
- [`anime_example`](anime_example.ipynb) - Media-content example
- [`ice concentration`](ice_concentration) - ice concentration example
- [`moving mnist example`](moving_mnist_example.ipynb) - MovingMnist dataset example


[`speed device test`](speed_device_test.py) - demonstrate time of building and device usage



Step-by-step examples description presented in ipynb cells and code comments.

## Common Issues
If you encounter the error ModuleNotFoundError: No module named 'torch':

1. Verify that the command pip list | grep torch confirms that torch is installed.

If torch appears in the command's output but the error persists, check that a Python virtual environment (venv) is created and used both for installing dependencies and when running your code.
You can create a virtual environment [as follows](https://docs.python.org/3/library/venv.html#creating-virtual-environments):

```
python -m venv venv
source venv/bin/activate
```
If the error continues, it is necessary to recreate the virtual environment from scratch and repeat the installation of TorchCNNBulder and its dependencies.

2. An error such as *"Minimum and Maximum cuda capability supported by this version of PyTorch is (7.0) - (12.0)"* can be caused by using an unsupported graphics card. TorchCNNBuidler only works with graphics cards that support CUDA versions >=7. In all other cases, operation is only possible in CPU mode.

To run the examples, you need to change the line device="cuda" to device="cpu".