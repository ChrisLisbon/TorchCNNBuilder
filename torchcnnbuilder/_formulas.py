from typing import Tuple

from torchcnnbuilder._utils import _double_params, _triple_params
from torchcnnbuilder._validation import _validate_conv_dim


# ------------------------------------
# calculation of convolution functions
# ------------------------------------
def conv1d_out(
    input_size: Tuple[int] | int,
    kernel_size: Tuple[int] | int = 3,
    stride: Tuple[int] | int = 1,
    padding: Tuple[int] | int = 0,
    dilation: Tuple[int] | int = 1,
    n_layers: int = 1,
) -> Tuple[int]:
    """
    Calculating the size of the tensor after nn.Conv1d

    :param input_size: size of the input tensor/vector [h]
    :param kernel_size: size of the convolution kernel. Default: 3
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param n_layers: number of conv layers
    :return int: size of the output tensor/vector [h] as a tuple
    # noqa
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
    input_size: Tuple[int] | int,
    kernel_size: Tuple[int] | int = 3,
    stride: Tuple[int] | int = 1,
    padding: Tuple[int] | int = 0,
    dilation: Tuple[int] | int = 1,
    n_layers: int = 1,
) -> Tuple[int, int]:
    """
    Calculating the size of the tensor after nn.Conv2d

    :param input_size: size of the input tensor [h, w]
    :param kernel_size: size of the convolution kernel. Default: 3
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param n_layers: number of conv layers
    :return Tuple[int, int]: size of the output tensor [h, w]
    # noqa
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
    input_size: Tuple[int] | int,
    kernel_size: Tuple[int] | int = 3,
    stride: Tuple[int] | int = 1,
    padding: Tuple[int] | int = 0,
    dilation: Tuple[int] | int = 1,
    n_layers: int = 1,
) -> Tuple[int, int, int]:
    """
    Calculating the size of the tensor after nn.Conv3d

    :param input_size: size of the input tensor [d, h, w]
    :param kernel_size: size of the convolution kernel. Default: 3
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param n_layers: number of conv layers
    :return Tuple[int, int, int]: size of the output tensor [d, h, w]
    # noqa
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
# calculation of transposed convolution functions
# ------------------------------------
def conv_transpose1d_out(
    input_size: Tuple[int] | int,
    kernel_size: Tuple[int] | int = 3,
    stride: Tuple[int] | int = 1,
    padding: Tuple[int] | int = 0,
    output_padding: Tuple[int] | int = 0,
    dilation: Tuple[int] | int = 1,
    n_layers: int = 1,
) -> Tuple[int]:
    """
    Calculating the size of the tensor after nn.ConvTranspose1d

    :param input_size: size of the input tensor/vector [h]
    :param kernel_size: size of the transposed convolution kernel. Default: 3
    :param stride: stride of the transposed convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param output_padding: controls the additional size added to one side of the output shape. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param n_layers: number of conv layers
    :return int: size of the output tensor/vector [h] as a tuple
    # noqa
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
    input_size: Tuple[int] | int,
    kernel_size: Tuple[int] | int = 3,
    stride: Tuple[int] | int = 1,
    padding: Tuple[int] | int = 0,
    output_padding: Tuple[int] | int = 0,
    dilation: Tuple[int] | int = 1,
    n_layers: int = 1,
) -> Tuple[int, int]:
    """
    Calculating the size of the tensor after nn.ConvTranspose2d

    :param input_size: size of the input tensor [h, w]
    :param kernel_size: size of the transposed convolution kernel. Default: 3
    :param stride: stride of the transposed convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param output_padding: controls the additional size added to one side of the output shape. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param n_layers: number of conv layers
    :return Tuple[int, int]: size of the output tensor [h, w]
    # noqa
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
    input_size: Tuple[int] | int,
    kernel_size: Tuple[int] | int = 3,
    stride: Tuple[int] | int = 1,
    padding: Tuple[int] | int = 0,
    output_padding: Tuple[int] | int = 0,
    dilation: Tuple[int] | int = 1,
    n_layers: int = 1,
) -> Tuple[int, int, int]:
    """
    Calculating the size of the tensor after nn.ConvTranspose3d

    :param input_size: size of the input tensor [d, h, w]
    :param kernel_size: size of the transposed convolution kernel. Default: 3
    :param stride: stride of the transposed convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param output_padding: controls the additional size added to one side of the output shape. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :param n_layers: number of conv layers
    :return Tuple[int, int]: size of the output tensor [d, h, w]
    # noqa
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


def _select_conv_calc(conv_dim: int, transpose: bool = False):
    """
    The function to select a way of calculating conv output

    :param conv_dim: the dimension of the convolutional operation
    :param transpose: choice of conv types between transposed and ordinary one. Default: False
    :return: one of functions to calculate conv or transposed conv output
    # noqa
    """
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
