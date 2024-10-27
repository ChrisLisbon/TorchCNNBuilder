# torchcnnbuilder

There are some functions for the calculation of the sizes of tensors after convolutional layers. 
Moreover, you can find here some base constants values

---
The size calculation after the convolutional layer is carried out according to the formula from the `torch` module *(the default parameters are the same as in `nn.ConvNd`)*. 
Counting functions are implemented for convolutions of dimensions from 1 to 3. 
At the same time, depending on the dimension, one number or the corresponding tuple of dimensions can be supplied to the parameters of each function. 
If it is necessary to calculate the convolution for tensors of N dimensions, then it is enough to simply apply a one-dimensional convolution N times. 
Some result values **can be negative** (due to the formula) which means you **should choose another conv params** (tensor dimensions degenerates to zero). 
The formula for calculating the size of the tensor after convolution for one dimension is presented below:

$$
H_{out} = \lfloor \frac{H_{in} + 2 \times padding[0] - dilation[0] \times (kernel[0] - 1) + 1}{stride[0]} \rfloor + 1
$$

## `conv1d_out`

Calculating the size of the tensor after `nn.Conv1d`
```python
def conv1d_out(
    input_size: Union[Tuple[int], int],
    kernel_size: Union[Tuple[int], int] = 3,
    stride: Union[Tuple[int], int] = 1,
    padding: Union[Tuple[int], int] = 0,
    dilation: Union[Tuple[int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int]:
```

**Params**:

- **input_size**: size of the input tensor/vector
- **kernel_size**: size of the convolution kernel. Default: 3
- **stride**: stride of the convolution. Default: 1
- **padding**: padding added to all four sides of the input. Default: 0
- **dilation**: spacing between kernel elements. Default: 1
- **n_layers**: number of conv layers

**Returns**: one tuple as a size of the output tensor

## `conv2d_out`

Calculating the size of the tensor after `nn.Conv2d`
```python
def conv2d_out(
    input_size: Union[Tuple[int, int], int],
    kernel_size: Union[Tuple[int, int], int] = 3,
    stride: Union[Tuple[int, int], int] = 1,
    padding: Union[Tuple[int, int], int] = 0,
    dilation: Union[Tuple[int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int]:
```

**Params**:

- **input_size**: size of the input tensor/vector
- **kernel_size**: size of the convolution kernel. Default: 3
- **stride**: stride of the convolution. Default: 1
- **padding**: padding added to all four sides of the input. Default: 0
- **dilation**: spacing between kernel elements. Default: 1
- **n_layers**: number of conv layers

**Returns**: one tuple as a size of the output tensor

## `conv3d_out`

Calculating the size of the tensor after `nn.Conv3d`
```python
def conv3d_out(
    input_size: Union[Tuple[int, int, int], int],
    kernel_size: Union[Tuple[int, int, int], int] = 3,
    stride: Union[Tuple[int, int, int], int] = 1,
    padding: Union[Tuple[int, int, int], int] = 0,
    dilation: Union[Tuple[int, int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int, int]:
```

**Params**:

- **input_size**: size of the input tensor/vector
- **kernel_size**: size of the convolution kernel. Default: 3
- **stride**: stride of the convolution. Default: 1
- **padding**: padding added to all four sides of the input. Default: 0
- **dilation**: spacing between kernel elements. Default: 1
- **n_layers**: number of conv layers

**Returns**: one tuple as a size of the output tensor

---

The size calculation after the transposed convolutional layer is carried out according to the formula from the torch module *(the default parameters are the same as in `nn.ConvTransposeNd`)*. 
Counting functions are implemented for transposed convolutions of dimensions from 1 to 3. 
At the same time, depending on the dimension, one number or the corresponding tuple of dimensions can be supplied to the parameters of each function. 
If it is necessary to calculate the transposed convolution for tensors of N dimensions, then it is enough to simply apply a one-dimensional transposed convolution N times. 
Some result values **can be negative** (due to the formula) which means you **should choose another conv params** (tensor dimensions degenerates to zero). 
The formula for calculating the size of the tensor after transposed convolution for one dimension is presented below:

$$
H_{out} = (H_{in} - 1) \times stride[0] - 2 \times padding[0] + dilation[0] \times (kernel\_size[0] - 1) + output\_padding[0] + 1
$$

## `conv_transpose1d_out`

Calculating the size of the tensor after `nn.ConvTranspose1d`

```python
def conv_transpose1d_out(
    input_size: Union[Tuple[int], int],
    kernel_size: Union[Tuple[int], int] = 3,
    stride: Union[Tuple[int], int] = 1,
    padding: Union[Tuple[int], int] = 0,
    output_padding: Union[Tuple[int], int] = 0,
    dilation: Union[Tuple[int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int]:
```

**Params**:

- **input_size**: size of the input tensor/vector
- **kernel_size**: size of the convolution kernel. Default: 3
- **stride**: stride of the convolution. Default: 1
- **padding**: padding added to all four sides of the input. Default: 0
- **output_padding**: controls the additional size added to one side of the output shape. Default: 0
- **dilation**: spacing between kernel elements. Default: 1
- **n_layers**: number of conv layers

**Returns**: one tuple as a size of the output tensor

## `conv_transpose2d_out`

Calculating the size of the tensor after `nn.ConvTranspose2d`

```python
def conv_transpose2d_out(
    input_size: Union[Tuple[int, int], int],
    kernel_size: Union[Tuple[int, int], int] = 3,
    stride: Union[Tuple[int, int], int] = 1,
    padding: Union[Tuple[int, int], int] = 0,
    output_padding: Union[Tuple[int, int], int] = 0,
    dilation: Union[Tuple[int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int]:
```

**Params**:

- **input_size**: size of the input tensor/vector
- **kernel_size**: size of the convolution kernel. Default: 3
- **stride**: stride of the convolution. Default: 1
- **padding**: padding added to all four sides of the input. Default: 0
- **output_padding**: controls the additional size added to one side of the output shape. Default: 0
- **dilation**: spacing between kernel elements. Default: 1
- **n_layers**: number of conv layers

**Returns**: one tuple as a size of the output tensor

## `conv_transpose3d_out`

Calculating the size of the tensor after `nn.ConvTranspose3d`

```python
def conv_transpose3d_out(
    input_size: Union[Tuple[int, int, int], int],
    kernel_size: Union[Tuple[int, int, int], int] = 3,
    stride: Union[Tuple[int, int, int], int] = 1,
    padding: Union[Tuple[int, int, int], int] = 0,
    output_padding: Union[Tuple[int, int, int], int] = 0,
    dilation: Union[Tuple[int, int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int, int]:
```

**Params**:

- **input_size**: size of the input tensor/vector
- **kernel_size**: size of the convolution kernel. Default: 3
- **stride**: stride of the convolution. Default: 1
- **padding**: padding added to all four sides of the input. Default: 0
- **output_padding**: controls the additional size added to one side of the output shape. Default: 0
- **dilation**: spacing between kernel elements. Default: 1
- **n_layers**: number of conv layers

**Returns**: one tuple as a size of the output tensor

--- 

# `DEFAULT_CONV_PARAMS`

You can check default torch convolution params:

```python
DEFAULT_CONV_PARAMS: Dict[str, int] = {
    "kernel_size": 3,
    "stride": 1,
    "padding": 0,
    "dilation": 1,
}
```

# `DEFAULT_TRANSPOSE_CONV_PARAMS`

You can check default torch transpose convolution params:

```
DEFAULT_TRANSPOSE_CONV_PARAMS: Dict[str, int] = {
    "kernel_size": 3,
    "stride": 1,
    "padding": 0,
    "output_padding": 0,
    "dilation": 1,
}
```
# `__version__`

You can check current package version by using constant `__version__`

```python
from torchcnnbuilder import __version__

print(__version__)
>>> 0.1.2
```




