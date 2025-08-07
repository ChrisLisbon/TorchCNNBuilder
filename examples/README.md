[[Click For Russian version]](README_RU.md)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TorchCNNBuilder.git
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