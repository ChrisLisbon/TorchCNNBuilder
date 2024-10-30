from typing import Tuple, Union

from torchcnnbuilder._utils import _double_params, _triple_params
from torchcnnbuilder._validation import _validate_conv_dim


# ------------------------------------
# Calculation of convolution functions
# ------------------------------------
def conv1d_out(
    input_size: Union[Tuple[int], int],
    kernel_size: Union[Tuple[int], int] = 3,
    stride: Union[Tuple[int], int] = 1,
    padding: Union[Tuple[int], int] = 0,
    dilation: Union[Tuple[int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int]:
    """
    Calculates the output size of a tensor after applying a 1D convolution operation (nn.Conv1d).

    The size calculation after the convolutional layer is carried out according to the formula from the `torch` module
    *(the default parameters are the same as in `nn.ConvNd`)*.
    Counting functions are implemented for convolutions of dimensions from 1 to 3. At the same time, depending on the
    dimension, one number or the corresponding tuple of dimensions can be supplied to the parameters of each function.
    If it is necessary to calculate the convolution for tensors of N dimensions, then it is enough to simply apply a
    one-dimensional convolution N times. Some result values **can be negative** (due to the formula) which means you
    **should choose another conv params** (tensor dimensions degenerates to zero). The formula for calculating the size
    of the tensor after convolution for one dimension is presented below:

    $$
    H_{out} = \lfloor \\frac{H_{in} + 2 \\times padding[0] -
    dilation[0] \\times (kernel[0] - 1) + 1}{stride[0]} \rfloor + 1
    $$

    Args:
        input_size (Union[Tuple[int], int]): Size of the input tensor or vector [h].
        kernel_size (Union[Tuple[int], int], optional): Size of the convolution kernel. Defaults to 3.
        stride (Union[Tuple[int], int], optional): Stride of the convolution. Defaults to 1.
        padding (Union[Tuple[int], int], optional): Padding added to both sides of the input. Defaults to 0.
        dilation (Union[Tuple[int], int], optional): Spacing between kernel elements. Defaults to 1.
        n_layers (int, optional): Number of convolutional layers. Defaults to 1.

    Returns:
        Tuple[int]: The size of the output tensor or vector [h] as a tuple.
    """
    input_size = (input_size,) if isinstance(input_size, int) else input_size
    padding = (padding,) if isinstance(padding, int) else padding
    dilation = (dilation,) if isinstance(dilation, int) else dilation
    kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    stride = (stride,) if isinstance(stride, int) else stride

    for _ in range(n_layers):
        h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        input_size = (int(h_out),)
    return input_size


def conv2d_out(
    input_size: Union[Tuple[int, int], int],
    kernel_size: Union[Tuple[int, int], int] = 3,
    stride: Union[Tuple[int, int], int] = 1,
    padding: Union[Tuple[int, int], int] = 0,
    dilation: Union[Tuple[int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int]:
    """
    Calculates the output size of a tensor after applying a 2D convolution operation (nn.Conv2d).

    Args:
        input_size (Union[Tuple[int, int], int]): Size of the input tensor [h, w].
        kernel_size (Union[Tuple[int, int], int], optional): Size of the convolution kernel. Defaults to 3.
        stride (Union[Tuple[int, int], int], optional): Stride of the convolution. Defaults to 1.
        padding (Union[Tuple[int, int], int], optional): Padding added to all sides of the input. Defaults to 0.
        dilation (Union[Tuple[int, int], int], optional): Spacing between kernel elements. Defaults to 1.
        n_layers (int, optional): Number of convolutional layers. Defaults to 1.

    Returns:
        Tuple[int, int]: The size of the output tensor [h, w].
    """
    input_size = _double_params(input_size) if isinstance(input_size, int) else input_size
    padding = _double_params(padding) if isinstance(padding, int) else padding
    dilation = _double_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _double_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _double_params(stride) if isinstance(stride, int) else stride

    for _ in range(n_layers):
        h_out = conv1d_out(input_size[0], kernel_size[0], stride[0], padding[0], dilation[0])[0]
        w_out = conv1d_out(input_size[1], kernel_size[1], stride[1], padding[1], dilation[1])[0]
        input_size = (int(h_out), int(w_out))
    return input_size


def conv3d_out(
    input_size: Union[Tuple[int, int, int], int],
    kernel_size: Union[Tuple[int, int, int], int] = 3,
    stride: Union[Tuple[int, int, int], int] = 1,
    padding: Union[Tuple[int, int, int], int] = 0,
    dilation: Union[Tuple[int, int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int, int]:
    """
    Calculates the output size of a tensor after applying a 3D convolution operation (nn.Conv3d).

    Args:
        input_size (Union[Tuple[int, int, int], int]): Size of the input tensor [d, h, w].
        kernel_size (Union[Tuple[int, int, int], int], optional): Size of the convolution kernel. Defaults to 3.
        stride (Union[Tuple[int, int, int], int], optional): Stride of the convolution. Defaults to 1.
        padding (Union[Tuple[int, int, int], int], optional): Padding added to all sides of the input. Defaults to 0.
        dilation (Union[Tuple[int, int, int], int], optional): Spacing between kernel elements. Defaults to 1.
        n_layers (int, optional): Number of convolutional layers. Defaults to 1.

    Returns:
        Tuple[int, int, int]: The size of the output tensor [d, h, w].
    """
    input_size = _triple_params(input_size) if isinstance(input_size, int) else input_size
    padding = _triple_params(padding) if isinstance(padding, int) else padding
    dilation = _triple_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _triple_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _triple_params(stride) if isinstance(stride, int) else stride

    for _ in range(n_layers):
        d_out = conv1d_out(input_size[0], kernel_size[0], stride[0], padding[0], dilation[0])[0]
        h_out = conv1d_out(input_size[1], kernel_size[1], stride[1], padding[1], dilation[1])[0]
        w_out = conv1d_out(input_size[2], kernel_size[2], stride[2], padding[2], dilation[2])[0]
        input_size = (int(d_out), int(h_out), int(w_out))
    return input_size


# ------------------------------------
# Calculation of transposed convolution functions
# ------------------------------------
def conv_transpose1d_out(
    input_size: Union[Tuple[int], int],
    kernel_size: Union[Tuple[int], int] = 3,
    stride: Union[Tuple[int], int] = 1,
    padding: Union[Tuple[int], int] = 0,
    output_padding: Union[Tuple[int], int] = 0,
    dilation: Union[Tuple[int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int]:
    """Calculates the output size of a tensor after a transposed 1D convolution (nn.ConvTranspose1d).

    The size calculation after the transposed convolutional layer is carried out according to the formula from the
    torch module *(the default parameters are the same as in `nn.ConvTransposeNd`)*. Counting functions are
    implemented for transposed convolutions of dimensions from 1 to 3. At the same time, depending on the dimension,
    one number or the corresponding tuple of dimensions can be supplied to the parameters of each function.
    If it is necessary to calculate the transposed convolution for tensors of N dimensions, then it is enough
    to simply apply a one-dimensional transposed convolution N times. Some result values **can be negative**
    (due to the formula) which means you **should choose another conv params** (tensor dimensions
    degenerates to zero). The formula for calculating the size of the tensor after transposed
    convolution for one dimension is presented below:

    $$
    H_{out} = (H_{in} - 1) \\times stride[0] - 2 \\times padding[0] + dilation[0]
    \\times (kernel\_size[0] - 1) + output\_padding[0] + 1
    $$

    Args:
        input_size (Union[Tuple[int], int]): Size of the input tensor/vector [h].
        kernel_size (Union[Tuple[int], int], optional): Size of the transposed convolution kernel. Defaults to 3.
        stride (Union[Tuple[int], int], optional): Stride of the transposed convolution. Defaults to 1.
        padding (Union[Tuple[int], int], optional): Padding added to both sides of the input. Defaults to 0.
        output_padding (Union[Tuple[int], int], optional): Additional size added to one side of the output shape.
        Defaults to 0.
        dilation (Union[Tuple[int], int], optional): Spacing between kernel elements. Defaults to 1.
        n_layers (int, optional): Number of convolutional layers. Defaults to 1.

    Returns:
        Tuple[int]: Size of the output tensor/vector [h].
    """
    input_size = (input_size,) if isinstance(input_size, int) else input_size
    padding = (padding,) if isinstance(padding, int) else padding
    output_padding = (output_padding,) if isinstance(output_padding, int) else output_padding
    dilation = (dilation,) if isinstance(dilation, int) else dilation
    kernel_size = (kernel_size,) if isinstance(kernel_size, int) else kernel_size
    stride = (stride,) if isinstance(stride, int) else stride

    for _ in range(n_layers):
        h_out = (
            (input_size[0] - 1) * stride[0]
            - 2 * padding[0]
            + dilation[0] * (kernel_size[0] - 1)
            + output_padding[0]
            + 1
        )
        input_size = (int(h_out),)
    return input_size


def conv_transpose2d_out(
    input_size: Union[Tuple[int, int], int],
    kernel_size: Union[Tuple[int, int], int] = 3,
    stride: Union[Tuple[int, int], int] = 1,
    padding: Union[Tuple[int, int], int] = 0,
    output_padding: Union[Tuple[int, int], int] = 0,
    dilation: Union[Tuple[int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int]:
    """Calculates the output size of a tensor after a transposed 2D convolution (nn.ConvTranspose2d).

    Args:
        input_size (Union[Tuple[int, int], int]): Size of the input tensor [h, w].
        kernel_size (Union[Tuple[int, int], int], optional): Size of the transposed convolution kernel. Defaults to 3.
        stride (Union[Tuple[int, int], int], optional): Stride of the transposed convolution. Defaults to 1.
        padding (Union[Tuple[int, int], int], optional): Padding added to both sides of the input. Defaults to 0.
        output_padding (Union[Tuple[int, int], int], optional): Additional size added to one side of the output shape.
        Defaults to 0.
        dilation (Union[Tuple[int, int], int], optional): Spacing between kernel elements. Defaults to 1.
        n_layers (int, optional): Number of convolutional layers. Defaults to 1.

    Returns:
        Tuple[int, int]: Size of the output tensor [h, w].
    """
    input_size = _double_params(input_size) if isinstance(input_size, int) else input_size
    padding = _double_params(padding) if isinstance(padding, int) else padding
    output_padding = _double_params(output_padding) if isinstance(output_padding, int) else output_padding
    dilation = _double_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _double_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _double_params(stride) if isinstance(stride, int) else stride

    for _ in range(n_layers):
        h_out = conv_transpose1d_out(
            input_size[0],
            kernel_size[0],
            stride[0],
            padding[0],
            output_padding[0],
            dilation[0],
        )[0]
        w_out = conv_transpose1d_out(
            input_size[1],
            kernel_size[1],
            stride[1],
            padding[1],
            output_padding[1],
            dilation[1],
        )[0]
        input_size = (int(h_out), int(w_out))
    return input_size


def conv_transpose3d_out(
    input_size: Union[Tuple[int, int, int], int],
    kernel_size: Union[Tuple[int, int, int], int] = 3,
    stride: Union[Tuple[int, int, int], int] = 1,
    padding: Union[Tuple[int, int, int], int] = 0,
    output_padding: Union[Tuple[int, int, int], int] = 0,
    dilation: Union[Tuple[int, int, int], int] = 1,
    n_layers: int = 1,
) -> Tuple[int, int, int]:
    """Calculates the output size of a tensor after a transposed 3D convolution (nn.ConvTranspose3d).

    Args:
        input_size (Union[Tuple[int, int, int], int]): Size of the input tensor [d, h, w].
        kernel_size (Union[Tuple[int, int, int], int], optional): Size of the transposed convolution kernel. Defaults to 3.
        stride (Union[Tuple[int, int, int], int], optional): Stride of the transposed convolution. Defaults to 1.
        padding (Union[Tuple[int, int, int], int], optional): Padding added to both sides of the input. Defaults to 0.
        output_padding (Union[Tuple[int, int, int], int], optional): Additional size added to one side of the output shape.
        Defaults to 0.
        dilation (Union[Tuple[int, int, int], int], optional): Spacing between kernel elements. Defaults to 1.
        n_layers (int, optional): Number of convolutional layers. Defaults to 1.

    Returns:
        Tuple[int, int, int]: Size of the output tensor [d, h, w].
    """
    input_size = _triple_params(input_size) if isinstance(input_size, int) else input_size
    padding = _triple_params(padding) if isinstance(padding, int) else padding
    output_padding = _triple_params(output_padding) if isinstance(output_padding, int) else output_padding
    dilation = _triple_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _triple_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _triple_params(stride) if isinstance(stride, int) else stride

    for _ in range(n_layers):
        d_out = conv_transpose1d_out(
            input_size[0],
            kernel_size[0],
            stride[0],
            padding[0],
            output_padding[0],
            dilation[0],
        )[0]
        h_out = conv_transpose1d_out(
            input_size[1],
            kernel_size[1],
            stride[1],
            padding[1],
            output_padding[1],
            dilation[1],
        )[0]
        w_out = conv_transpose1d_out(
            input_size[2],
            kernel_size[2],
            stride[2],
            padding[2],
            output_padding[2],
            dilation[1],
        )[0]
        input_size = (int(d_out), int(h_out), int(w_out))
    return input_size


# ------------------------------------
# Select formulas functions
# ------------------------------------
def _select_conv_calc(conv_dim: int, transpose: bool = False):
    _validate_conv_dim(conv_dim)

    if conv_dim == 1:
        if transpose:
            return conv_transpose1d_out
        return conv1d_out

    if conv_dim == 2:
        if transpose:
            return conv_transpose2d_out
        return conv2d_out

    if conv_dim == 3:
        if transpose:
            return conv_transpose3d_out
        return conv3d_out
