from typing import Union, Sequence, Tuple


def _double_params(param: int) -> Tuple[int, int]:
    """
    :param param: int param of some function
    :return Tuple[int, int]: doubled param
    """
    return param, param


def conv2d_out(input_size: Union[Sequence[int], int],
               kernel_size: Union[Sequence[int], int],
               stride: Union[Sequence[int], int] = 1,
               padding: Union[Sequence[int], int] = 0,
               dilation: Union[Sequence[int], int] = 1) -> Tuple[int, int]:
    """
    :param input_size: size of the input tensor [h, w]
    :param kernel_size: size of the convolving kernel
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :return Tuple[int, int]: size of the output tensor [h, w]
    """
    input_size = _double_params(input_size) if isinstance(input_size, int) else input_size
    padding = _double_params(padding) if isinstance(padding, int) else padding
    dilation = _double_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _double_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _double_params(stride) if isinstance(stride, int) else stride

    h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    return int(h_out), int(w_out)


def conv_transpose2d_out(input_size: Union[Sequence[int], int],
                         kernel_size: Union[Sequence[int], int],
                         stride: Union[Sequence[int], int] = 1,
                         padding: Union[Sequence[int], int] = 0,
                         output_padding: Union[Sequence[int], int] = 0,
                         dilation: Union[Sequence[int], int] = 1) -> Tuple[int, int]:
    """
    :param input_size: size of the input tensor [h, w]
    :param kernel_size: size of the convolving kernel
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param output_padding: controls the additional size added to one side of the output shape. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :return Tuple[int, int]: size of the output tensor [h, w]
    """
    input_size = _double_params(input_size) if isinstance(input_size, int) else input_size
    padding = _double_params(padding) if isinstance(padding, int) else padding
    output_padding = _double_params(output_padding) if isinstance(output_padding, int) else output_padding
    dilation = _double_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _double_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _double_params(stride) if isinstance(stride, int) else stride

    h_out = (input_size[0] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[
        0] + 1
    w_out = (input_size[1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[
        1] + 1
    return int(h_out), int(w_out)