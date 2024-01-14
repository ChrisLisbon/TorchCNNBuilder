from typing import Union, Sequence, Tuple, Optional
from collections import OrderedDict

import torch.nn as nn


def _double_params(param: int) -> Tuple[int, int]:
    """
    :param param: int param of some function
    :return Tuple[int, int]: doubled param
    """
    return param, param


def _triple_params(param: int) -> Tuple[int, int, int]:
    """
    :param param: int param of some function
    :return Tuple[int, int, int]: tripled param
    """
    return param, param, param


# ------------------------------------
# calculation of convolution functions
# ------------------------------------
def conv1d_out(input_size: int,
               kernel_size: int,
               stride: int = 1,
               padding: int = 0,
               dilation: int = 1) -> int:
    """
    :param input_size: size of the input tensor/vector [h]
    :param kernel_size: size of the convolution kernel
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :return int: size of the output tensor/vector [h]
    """
    h_out = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    return int(h_out)


def conv2d_out(input_size: Union[Sequence[int], int],
               kernel_size: Union[Sequence[int], int],
               stride: Union[Sequence[int], int] = 1,
               padding: Union[Sequence[int], int] = 0,
               dilation: Union[Sequence[int], int] = 1) -> Tuple[int, int]:
    """
    :param input_size: size of the input tensor [h, w]
    :param kernel_size: size of the convolution kernel
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

    h_out = conv1d_out(input_size[0], kernel_size[0], stride[0], padding[0], dilation[0])
    w_out = conv1d_out(input_size[1], kernel_size[1], stride[1], padding[1], dilation[1])
    return int(h_out), int(w_out)


def conv3d_out(input_size: Union[Sequence[int], int],
               kernel_size: Union[Sequence[int], int],
               stride: Union[Sequence[int], int] = 1,
               padding: Union[Sequence[int], int] = 0,
               dilation: Union[Sequence[int], int] = 1) -> Tuple[int, int, int]:
    """
    :param input_size: size of the input tensor [d, h, w]
    :param kernel_size: size of the convolution kernel
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :return Tuple[int, int, int]: size of the output tensor [d, h, w]
    """
    input_size = _triple_params(input_size) if isinstance(input_size, int) else input_size
    padding = _triple_params(padding) if isinstance(padding, int) else padding
    dilation = _triple_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _triple_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _triple_params(stride) if isinstance(stride, int) else stride

    d_out = conv1d_out(input_size[0], kernel_size[0], stride[0], padding[0], dilation[0])
    h_out = conv1d_out(input_size[1], kernel_size[1], stride[1], padding[1], dilation[1])
    w_out = conv1d_out(input_size[2], kernel_size[2], stride[2], padding[2], dilation[2])
    return int(d_out), int(h_out), int(w_out)


# ------------------------------------
# calculation of transpose convolution functions
# ------------------------------------
def conv_transpose1d_out(input_size: int,
                         kernel_size: int,
                         stride: int = 1,
                         padding: int = 0,
                         output_padding: int = 0,
                         dilation: int = 1) -> int:
    """
    :param input_size: size of the input tensor/vector [h]
    :param kernel_size: size of the convolution kernel
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param output_padding: controls the additional size added to one side of the output shape. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :return int: size of the output tensor/vector [h]
    """
    h_out = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return int(h_out)


def conv_transpose2d_out(input_size: Union[Sequence[int], int],
                         kernel_size: Union[Sequence[int], int],
                         stride: Union[Sequence[int], int] = 1,
                         padding: Union[Sequence[int], int] = 0,
                         output_padding: Union[Sequence[int], int] = 0,
                         dilation: Union[Sequence[int], int] = 1) -> Tuple[int, int]:
    """
    :param input_size: size of the input tensor [h, w]
    :param kernel_size: size of the convolution kernel
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

    h_out = conv_transpose1d_out(input_size[0], kernel_size[0], stride[0], padding[0], output_padding[0], dilation[0])
    w_out = conv_transpose1d_out(input_size[1], kernel_size[1], stride[1], padding[1], output_padding[1], dilation[1])
    return int(h_out), int(w_out)


def conv_transpose3d_out(input_size: Union[Sequence[int], int],
                         kernel_size: Union[Sequence[int], int],
                         stride: Union[Sequence[int], int] = 1,
                         padding: Union[Sequence[int], int] = 0,
                         output_padding: Union[Sequence[int], int] = 0,
                         dilation: Union[Sequence[int], int] = 1) -> Tuple[int, int, int]:
    """
    :param input_size: size of the input tensor [d, h, w]
    :param kernel_size: size of the convolution kernel
    :param stride: stride of the convolution. Default: 1
    :param padding: padding added to all four sides of the input. Default: 0
    :param output_padding: controls the additional size added to one side of the output shape. Default: 0
    :param dilation: spacing between kernel elements. Default: 1
    :return Tuple[int, int]: size of the output tensor [d, h, w]
    """
    input_size = _triple_params(input_size) if isinstance(input_size, int) else input_size
    padding = _triple_params(padding) if isinstance(padding, int) else padding
    output_padding = _triple_params(output_padding) if isinstance(output_padding, int) else output_padding
    dilation = _triple_params(dilation) if isinstance(dilation, int) else dilation
    kernel_size = _triple_params(kernel_size) if isinstance(kernel_size, int) else kernel_size
    stride = _triple_params(stride) if isinstance(stride, int) else stride

    d_out = conv_transpose1d_out(input_size[0], kernel_size[0], stride[0], padding[0], output_padding[0], dilation[0])
    h_out = conv_transpose1d_out(input_size[1], kernel_size[1], stride[1], padding[1], output_padding[1], dilation[1])
    w_out = conv_transpose1d_out(input_size[2], kernel_size[2], stride[2], padding[2], output_padding[2], dilation[1])
    return int(d_out), int(h_out), int(w_out)


# ------------------------------------
# CNN Builder class
# ------------------------------------
class EncoderBuilder:
    def __init__(self,
                 input_size: Sequence[int],
                 minimum_feature_map_size: Union[Sequence[int], int] = 5):

        self.input_size = input_size
        #self.n_vectors = n_vectors

        self.minimum_feature_map_size = _double_params(minimum_feature_map_size) \
            if isinstance(minimum_feature_map_size, int) \
            else minimum_feature_map_size

        self.max_channels = 512
        self.min_channels = 32

        self.default_convolve_params = {'kernel_size': 3,
                                        'stride': 1,
                                        'padding': 0,
                                        'dilation': 1
                                        }

        self.default_transpose_params = {'kernel_size': 3,
                                         'stride': 1,
                                         'padding': 0,
                                         'output_padding': 0,
                                         'dilation': 1
                                         }

        self.activation_function = nn.ReLU(inplace=True)
        self.finish_activation_function = None

        self.input_channels_count_list = None
        self.input_layer_size_list = None

    def build_convolve_block(self,
                             in_channels: int,
                             out_channels: int,
                             params: Optional[dict] = None,
                             normalization: Optional[str] = None,
                             sub_blocks: int = 1,
                             p: float = 0.5,
                             inplace: bool = False,
                             eps: float = 1e-5,
                             momentum: Optional[float] = 0.1,
                             affine: bool = True) -> nn.Sequential:
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution
        :param params: convolutional layer parameters (nn.Conv2d). Default: None
        :param normalization: choice of normalization between str 'dropout' and 'batchnorm'. Default: None
        :param sub_blocks: number of convolutions in one layer. Default: 1
        :param p: probability of an element to be zero-ed. Default: 0.5
        :param inplace: if set to True, will do this operation in-place. Default: False
        :param eps: a value added to the denominator for numerical stability. Default: 1e-5
        :param momentum: used for the running_mean -_var computation. Can be None for cumulative moving average. Default: 0.1
        :param affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
        :return nn.Sequential: one convolution block with an activation function
        """

        default_convolve_params = self.default_convolve_params.copy()

        if params:
            for key, value in params.items():
                value = _double_params(value) if isinstance(value, int) else value
                default_convolve_params[key] = value

        params = default_convolve_params

        if sub_blocks > 1:
            kernel_size = default_convolve_params['kernel_size']
            kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            default_convolve_params['padding'] = kernel_size // 2
            default_convolve_params['stride'] = 1

        blocks = []
        for i in range(sub_blocks):
            block = []

            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             **params)
            in_channels = out_channels

            block.append(conv)

            if normalization == 'batchnorm':
                norm = nn.BatchNorm2d(num_features=out_channels,
                                      eps=eps,
                                      momentum=momentum,
                                      affine=affine)
                block.append(norm)

            if normalization == 'dropout':
                norm = nn.Dropout2d(p=p,
                                    inplace=inplace)
                block.append(norm)

            activation_function = self.activation_function
            block.append(activation_function)

            if sub_blocks > 1:
                block = nn.Sequential(*block)
                blocks.append((f'sub-block {i+1}', block))
            else:
                blocks.extend(block)

        if sub_blocks > 1:
            return nn.Sequential(OrderedDict(blocks))

        return nn.Sequential(*blocks)

    def build_convolve_sequence(self,
                                n_layers: int,
                                in_channels: int = 1,
                                params: Optional[dict] = None,
                                normalization: Optional[str] = None,
                                sub_blocks: int = 1,
                                up: float = 2.0,
                                low: float = 1.5,
                                ascending: bool = False) -> nn.Sequential:
        """
        :param n_layers: number of the convolution layers in the encoder part
        :param in_channels: number of channels in the first input tensor. Default: 1
        :param params: convolutional layer parameters (nn.Conv2d). Default: None
        :param normalization: choice of normalization between str 'dropout' and 'batchnorm'. Default: None
        :param sub_blocks: number of convolutions in one layer. Default: 1
        :param up: multiplier for the geometric progression of increasing channels (feature maps). Default: 2 (powers of two)
        :param low: multiplier for the geometric progression of decreasing channels (feature maps). Default: 1.5
        :param ascending: the way of calculating the number of feature maps (with using 'up' or 'low'). Default: False
        :return nn.Sequential: convolutional sequence
        """

        default_convolve_params = self.default_convolve_params.copy()

        if params:
            for key, value in params.items():
                value = _double_params(value) if isinstance(value, int) else value
                default_convolve_params[key] = value

        params = default_convolve_params

        modules = []
        input_layer_size_list = [self.input_size]
        input_channels_count_list = [in_channels]

        for layer in range(n_layers):

            input_layer_size = input_layer_size_list[-1]

            if (input_layer_size[0] < self.minimum_feature_map_size[0]
                    and input_layer_size[1] < self.minimum_feature_map_size[1]):

                raise ValueError(f'Input size and parameters can not provide more than {layer + 1} layers')

            elif input_channels_count_list[-1] > self.max_channels:
                raise ValueError(f'There is too many channels. Max channels {self.max_channels}')

            else:

                in_channels = input_channels_count_list[-1]
                out_channels = self._calc_out_channels(input_size=input_layer_size,
                                                       input_channels=in_channels,
                                                       layer=layer,
                                                       up=up,
                                                       low=low,
                                                       ascending=ascending)

                input_channels_count_list.append(out_channels)
                out_layer_size = conv2d_out(input_layer_size, **params)
                input_layer_size_list.append(out_layer_size)

                convolve_block = self.build_convolve_block(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           normalization=normalization,
                                                           sub_blocks=sub_blocks,
                                                           params=params)

                modules.append((f'conv {layer+1}', convolve_block))

        self.input_channels_count_list = input_channels_count_list
        self.input_layer_size_list = input_layer_size_list
        return nn.Sequential(OrderedDict(modules))

    def _calc_out_channels(self,
                           input_size: Sequence[int],
                           input_channels: int,
                           layer: int,
                           up: float = 2.0,
                           low: float = 1.5,
                           ascending: bool = False) -> int:
        if ascending:
            return int((input_size[0] + input_size[1]) / low)

        if layer == 0:
            return self.min_channels
        return int(input_channels * up)