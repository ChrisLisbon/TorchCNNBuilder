import math
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple, Union

import torch.nn as nn

from torchcnnbuilder._constants import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_TRANSPOSE_CONV_PARAMS,
)
from torchcnnbuilder._formulas import _select_conv_calc
from torchcnnbuilder._utils import (
    _double_params,
    _select_adaptive_pooling_dimension,
    _select_conv_dimension,
    _select_norm_dimension,
    _set_conv_params,
    _triple_params,
)
from torchcnnbuilder._validation import (
    _validate_available_layers,
    _validate_build_transpose_convolve_init,
    _validate_channel_growth_rate_param,
    _validate_difference_in_dimensions,
    _validate_input_size_is_not_none,
    _validate_max_channels_number,
    _validate_min_channels_number,
    _validate_range_step,
)
from torchcnnbuilder.latent import LatentSpaceModule


# ------------------------------------
# CNN Builder class
# ------------------------------------
class Builder:
    """
    A class for creating СNN architectures

    Attributes:
        input_size (Sequence[int]): input size of the input tensor
        minimum_feature_map_size (Union[Tuple, int]): minimum feature map size. Default: 5
        max_channels (int): maximum number of layers after any convolution. Default: 512
        min_channels (int): minimum number of layers after any convolution. Default: 32
        activation_function (nn.Module): activation function. Default: nn.ReLU(inplace=True)
        finish_activation_function (Union[str, Optional[nn.Module]): last activation function, can be same as activation_function (str 'same'). Default: None
        conv_channels (List[int]): list of output channels after each convolutional layer
        transpose_conv_channels (List[int]): list of output channels after each transposed convolutional layer
        conv_layers (List[tuple]): list of output tensor sizes after each convolutional layer
        transpose_conv_layers (List[tuple]): list of output tensor sizes after each transposed convolutional layer
        # noqa
    """

    def __init__(
        self,
        input_size: Optional[Sequence[int]] = None,
        minimum_feature_map_size: Union[Sequence[int], int] = 5,
        max_channels: int = 512,
        min_channels: int = 1,
        activation_function: nn.Module = nn.ReLU(inplace=True),
        finish_activation_function: Union[Optional[nn.Module], str] = None,
    ) -> None:
        """
        The constructor for Builder

        :param input_size: input size of the input tensor. Necessary for creating conv sequences. Default: None
        :param minimum_feature_map_size: minimum feature map size. Default: 5
        :param max_channels: maximum number of layers after any convolution. Default: 512
        :param min_channels: minimum number of layers after any convolution. Default: 32
        :param activation_function: activation function. Default: nn.ReLU(inplace=True)
        :param finish_activation_function: last activation function, can be same as activation_function (str 'same'). Default: None
        # noqa
        """

        if input_size is None:
            self.input_size = input_size
        else:
            self.input_size = tuple(i for i in input_size)

            if len(self.input_size) == 1:
                self.minimum_feature_map_size = (
                    (minimum_feature_map_size,)
                    if isinstance(minimum_feature_map_size, int)
                    else minimum_feature_map_size
                )
            if len(self.input_size) == 2:
                self.minimum_feature_map_size = (
                    _double_params(
                        minimum_feature_map_size,
                    )
                    if isinstance(minimum_feature_map_size, int)
                    else minimum_feature_map_size
                )
            if len(self.input_size) == 3:
                self.minimum_feature_map_size = (
                    _triple_params(
                        minimum_feature_map_size,
                    )
                    if isinstance(minimum_feature_map_size, int)
                    else minimum_feature_map_size
                )

        self.max_channels = max_channels
        self._initial_max_channels = max_channels

        self.min_channels = min_channels
        self._initial_min_channels = min_channels

        self._default_convolve_params = DEFAULT_CONV_PARAMS
        self._default_transpose_params = DEFAULT_TRANSPOSE_CONV_PARAMS

        # finish_activation_function can be str 'same' which equals to activation_function
        self.activation_function = activation_function
        self.finish_activation_function = finish_activation_function

        self._conv_channels = None
        self._transpose_conv_channels = None

        self._conv_layers = None
        self._transpose_conv_layers = None

    @property
    def conv_channels(self) -> Optional[List[int]]:
        return self._conv_channels

    @property
    def transpose_conv_channels(self) -> Optional[List[int]]:
        return self._transpose_conv_channels

    @property
    def conv_layers(self) -> Optional[List[Tuple[int, ...]]]:
        return self._conv_layers

    @property
    def transpose_conv_layers(self) -> Optional[List[Tuple[int, ...]]]:
        return self._transpose_conv_layers

    def build_convolve_block(
        self,
        in_channels: int,
        out_channels: int,
        params: Optional[dict] = None,
        normalization: Optional[str] = None,
        sub_blocks: int = 1,
        p: float = 0.5,
        inplace: bool = False,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        conv_dim: int = 2,
    ) -> nn.Sequential:
        """
        The function to build a single block of convolution layers

        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution
        :param params: convolutional layer parameters (nn.ConvNd). Default: None
        :param normalization: choice of normalization between str 'dropout', 'instancenorm' and 'batchnorm'. Default: None
        :param sub_blocks: number of convolutions in one layer. Default: 1
        :param p: probability of an element to be zero-ed (for dropout/instancenorm). Default: 0.5
        :param inplace: if set to True, will do this operation in-place (for dropout/instancenorm). Default: False
        :param eps: a value added to the denominator for numerical stability (for batchnorm/instancenorm). Default: 1e-5
        :param momentum: used for the running_mean or running_var computation. Can be None for cumulative moving average (for batchnorm). Default: 0.1
        :param affine: a boolean value, when set to True, this module has learnable affine parameters (for batchnorm). Default: True
        :param conv_dim: the dimension of the convolutional operation. Default: 2
        :return nn.Sequential: one convolution block with an activation function
        # noqa
        """
        params = _set_conv_params(default_params=self._default_convolve_params, params=params)
        convolution = _select_conv_dimension(conv_dim=conv_dim)

        if sub_blocks > 1:
            kernel_size = params["kernel_size"]
            kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            params["padding"] = kernel_size // 2
            params["stride"] = 1

        blocks = []
        for i in range(sub_blocks):
            block = []

            conv = convolution(in_channels=in_channels, out_channels=out_channels, **params)
            in_channels = out_channels
            block.append(conv)

            if normalization:
                norm = _select_norm_dimension(conv_dim=conv_dim, normalization=normalization)

                if normalization in ("batchnorm", "instancenorm"):
                    norm = norm(
                        num_features=out_channels,
                        eps=eps,
                        momentum=momentum,
                        affine=affine,
                    )

                if normalization == "dropout":
                    norm = norm(p=p, inplace=inplace)

                block.append(norm)

            activation_function = self.activation_function
            block.append(activation_function)

            if sub_blocks > 1:
                block = nn.Sequential(*block)
                blocks.append((f"sub-block {i + 1}", block))
            else:
                blocks.extend(block)

        if sub_blocks > 1:
            return nn.Sequential(OrderedDict(blocks))

        return nn.Sequential(*blocks)

    def build_convolve_sequence(
        self,
        n_layers: int,
        in_channels: int = 1,
        params: Optional[dict] = None,
        normalization: Optional[str] = None,
        sub_blocks: int = 1,
        p: float = 0.5,
        inplace: bool = False,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        ratio: float = 2.0,
        start: int = 32,
        channel_growth_rate: str = "exponential",
        conv_dim: int = 2,
    ) -> nn.Sequential:
        """
         The function to build a sequence of convolution blocks

        :param n_layers: number of the convolution layers in the encoder part
        :param in_channels: number of channels in the first input tensor. Default: 1
        :param params: convolutional layer parameters (nn.ConvNd). Default: None
        :param normalization: choice of normalization between str 'dropout', 'instancenorm' and 'batchnorm'. Default: None
        :param sub_blocks: number of convolutions in one layer. Default: 1
        :param p: probability of an element to be zero-ed (for dropout). Default: 0.5
        :param inplace: if set to True, will do this operation in-place (for dropout). Default: False
        :param eps: a value added to the denominator for numerical stability (for batchnorm/instancenorm). Default: 1e-5
        :param momentum: used for the running_mean or running_var computation. Can be None for cumulative moving average (for batchnorm/instancenorm). Default: 0.1
        :param affine: a boolean value, when set to True, this module has learnable affine parameters (for batchnorm/instancenorm). Default: True
        :param ratio: multiplier for the geometric progression of increasing channels (feature maps). Used for 'channel_growth_rate' as 'exponential' or 'power' . Default: 2 (powers of two)
        :param start: start position of a geometric progression in the case of 'channel_growth_rate=exponential'. Default: 32
        :param channel_growth_rate: the way of calculating the number of feature maps between 'exponential', 'proportion', 'linear', 'power' and 'constant'. Default: 'exponential'
        :param conv_dim: the dimension of the convolutional operation. Default: 2
        :return nn.Sequential: convolutional sequence
        # noqa
        """
        _validate_input_size_is_not_none(self.input_size)
        params = _set_conv_params(default_params=self._default_convolve_params, params=params)
        conv_out = _select_conv_calc(conv_dim=conv_dim)

        modules = []
        input_layer_size_list = [self.input_size]
        input_channels_count_list = self._calc_out_channels(
            in_size=self.input_size,
            in_channels=in_channels,
            n_layers=n_layers,
            ratio=ratio,
            start=start,
            channel_growth_rate=channel_growth_rate,
        )

        for layer in range(n_layers):
            input_layer_size = input_layer_size_list[-1]

            _validate_difference_in_dimensions(self.input_size, conv_dim)
            _validate_available_layers(layer, input_layer_size, self.minimum_feature_map_size)
            _validate_max_channels_number(layer, input_channels_count_list, self.max_channels)
            _validate_min_channels_number(layer, input_channels_count_list, self.min_channels)

            in_channels = input_channels_count_list[layer]
            out_channels = input_channels_count_list[layer + 1]

            out_layer_size = conv_out(input_size=input_layer_size, **params)
            input_layer_size_list.append(out_layer_size)

            convolve_block = self.build_convolve_block(
                in_channels=in_channels,
                out_channels=out_channels,
                normalization=normalization,
                sub_blocks=sub_blocks,
                p=p,
                inplace=inplace,
                eps=eps,
                momentum=momentum,
                affine=affine,
                params=params,
                conv_dim=conv_dim,
            )

            modules.append((f"conv {layer + 1}", convolve_block))

        self._conv_channels = input_channels_count_list
        self._conv_layers = input_layer_size_list
        return nn.Sequential(OrderedDict(modules))

    def build_transpose_convolve_block(
        self,
        in_channels: int,
        out_channels: int,
        params: Optional[dict] = None,
        normalization: Optional[str] = None,
        sub_blocks: int = 1,
        p: float = 0.5,
        inplace: bool = False,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        last_block: bool = False,
        conv_dim: int = 2,
    ) -> nn.Sequential:
        """
        The function to build a single block of transposed convolution layers

        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution
        :param params: convolutional layer parameters (nn.Conv2d). Default: None
        :param normalization: choice of normalization between str 'dropout', 'instancenorm' and 'batchnorm'. Default: None
        :param sub_blocks: number of convolutions in one layer. Default: 1
        :param p: probability of an element to be zero-ed (for dropout). Default: 0.5
        :param inplace: if set to True, will do this operation in-place (for dropout). Default: False
        :param eps: a value added to the denominator for numerical stability (for batchnorm/instancenorm). Default: 1e-5
        :param momentum: used for the running_mean or running_var computation. Can be None for cumulative moving average (for batchnorm/instancenorm). Default: 0.1
        :param affine: a boolean value, when set to True, this module has learnable affine parameters (for batchnorm/instancenorm). Default: True
        :param last_block: if True there is no activation function after the transposed convolution. Default: False
        :param conv_dim: the dimension of the convolutional operation. Default: 2
        :return nn.Sequential: one convolution block with an activation function
        # noqa
        """
        params = _set_conv_params(default_params=self._default_transpose_params, params=params)
        convolution = _select_conv_dimension(conv_dim=conv_dim, transpose=True)

        if sub_blocks > 1:
            kernel_size = params["kernel_size"]
            kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            params["padding"] = kernel_size // 2
            params["stride"] = 1

        blocks = []
        last_out_channels = out_channels
        for i in range(sub_blocks):
            block = []

            out_channels = last_out_channels if i == sub_blocks - 1 else in_channels
            conv = convolution(in_channels=in_channels, out_channels=out_channels, **params)
            block.append(conv)

            if normalization:
                norm = _select_norm_dimension(conv_dim=conv_dim, normalization=normalization)

                if normalization in ("batchnorm", "instancenorm"):
                    norm = norm(
                        num_features=out_channels,
                        eps=eps,
                        momentum=momentum,
                        affine=affine,
                    )

                if normalization == "dropout":
                    norm = norm(p=p, inplace=inplace)

                block.append(norm)

            activation_function = self.activation_function
            if last_block and i == sub_blocks - 1:
                if self.finish_activation_function == "same":
                    block.append(activation_function)
                elif self.finish_activation_function:
                    block.append(self.finish_activation_function)
            else:
                block.append(activation_function)

            if sub_blocks > 1:
                block = nn.Sequential(*block)
                blocks.append((f"transpose sub-block {i + 1}", block))
            else:
                blocks.extend(block)

        if sub_blocks > 1:
            return nn.Sequential(OrderedDict(blocks))

        return nn.Sequential(*blocks)

    def build_transpose_convolve_sequence(
        self,
        n_layers: int,
        in_channels: Optional[int] = None,
        out_channels: int = 1,
        out_size: Optional[tuple] = None,
        params: Optional[dict] = None,
        normalization: Optional[str] = None,
        sub_blocks: int = 1,
        p: float = 0.5,
        inplace: bool = False,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        ratio: float = 2.0,
        channel_growth_rate: str = "exponential",
        conv_dim: int = 2,
        adaptive_pool: str = "avgpool",
    ) -> nn.Sequential:
        """
        The function to build a sequence of transposed convolution blocks

        :param n_layers: number of the convolution layers in the encoder part
        :param in_channels: number of channels in the first input tensor. Default: None
        :param out_channels: number of channels after the transposed convolution sequence. Default: 1
        :param out_size: output size after the transposed convolution sequence. Default: None (input size)
        :param params: transposed convolutional layer parameters (nn.ConvTranspose2d). Default: None
        :param normalization: choice of normalization between str 'dropout', 'instancenorm' and 'batchnorm'. Default: None
        :param sub_blocks: number of transposed convolutions in one layer. Default: 1
        :param p: probability of an element to be zero-ed (for dropout). Default: 0.5
        :param inplace: if set to True, will do this operation in-place (for dropout). Default: False
        :param eps: a value added to the denominator for numerical stability (for batchnorm/instancenorm). Default: 1e-5
        :param momentum: used for the running_mean or runnIng_var computation. Can be None for cumulative moving average (for batchnorm/instancenorm). Default: 0.1
        :param affine: a boolean value, when set to True, this module has learnable affine parameters (for batchnorm/instancenorm). Default: True
        :param ratio: multiplier for the geometric progression of increasing channels (feature maps). Used for 'channel_growth_rate' as 'exponential' or 'power' . Default: 2 (powers of two)
        :param channel_growth_rate: the way of calculating the number of feature maps between 'exponential', 'proportion', 'linear', 'power' and 'constant'. Default: 'exponential'
        :param conv_dim: the dimension of the convolutional operation. Default: 2
        :param adaptive_pool: choice of a last layer as an adaptive pooling between str 'avgpool' or 'maxpool'. Default: 'avgpool'
        :return nn.Sequential: transposed convolutional sequence
        # noqa
        """
        _validate_input_size_is_not_none(self.input_size)
        params = _set_conv_params(default_params=self._default_transpose_params, params=params)
        conv_out = _select_conv_calc(conv_dim=conv_dim, transpose=True)

        modules = []

        if in_channels is None and self._conv_channels:
            in_channels = self._conv_channels[-1]

        _validate_build_transpose_convolve_init(in_channels, self._conv_channels)

        if self._conv_layers:
            input_layer_size_list = [self._conv_layers[-1]]

        input_channels_count_list = self._calc_out_transpose_channels(
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            ratio=ratio,
            channel_growth_rate=channel_growth_rate,
        )
        for layer in range(n_layers):
            _validate_max_channels_number(layer, input_channels_count_list, self.max_channels)
            _validate_min_channels_number(layer, input_channels_count_list, min_channels=1)

            in_channels = input_channels_count_list[layer]
            out_channels = input_channels_count_list[layer + 1]

            if self._conv_layers:
                input_layer_size = input_layer_size_list[-1]
                out_layer_size = conv_out(input_size=input_layer_size, **params)
                input_layer_size_list.append(out_layer_size)

            last_block_condition = layer == n_layers - 1
            convolve_block = self.build_transpose_convolve_block(
                in_channels=in_channels,
                out_channels=out_channels,
                normalization=normalization,
                sub_blocks=sub_blocks,
                p=p,
                inplace=inplace,
                eps=eps,
                momentum=momentum,
                affine=affine,
                params=params,
                last_block=last_block_condition,
                conv_dim=conv_dim,
            )

            modules.append((f"deconv {layer + 1}", convolve_block))

        self._transpose_conv_channels = input_channels_count_list

        if self._conv_layers:
            self._transpose_conv_layers = input_layer_size_list

        if out_size is None:
            out_size = self.input_size

        adaptive_pooling = _select_adaptive_pooling_dimension(conv_dim=conv_dim, pooling=adaptive_pool)
        resize_block = adaptive_pooling(output_size=tuple(out_size))
        modules.append(("resize", resize_block))

        return nn.Sequential(OrderedDict(modules))

    def latent_block(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        n_layers: int = 1,
        activation: bool = False,
        activation_function: Optional[nn.Module] = None,
    ):
        """
        Creates a latent space transformation block.

        :param input_shape: the shape of the input tensor
        :param output_shape: the desired shape of the output tensor
        :param n_layers: number of linear layers to use in the transformation. Default: 1
        :param activation: whether to apply an activation function after each linear layer. Default: False
        :param activation_function: if not provided, defaults to the module's activation function. Default: None
        # noqa
        """
        if activation_function is None:
            activation_function = self.activation_function

        return LatentSpaceModule(input_shape, output_shape, n_layers, activation, activation_function)

    def _calc_out_channels(
        self,
        in_size: Sequence[int],
        in_channels: int,
        n_layers: int,
        ratio: float = 2.0,
        start: int = 32,
        constant: int = 1,
        channel_growth_rate: str = "exponential",
    ) -> List[int]:
        """
        The function to calculate output channels after each convolutional layer

        :param in_size: input size of the first input tensor
        :param in_channels: number of channels in the first input tensor
        :param n_layers: number of the convolution layers in the encoder part
        :param ratio: multiplier for the geometric progression of increasing channels (feature maps). Used for 'channel_growth_rate' as 'exponential' or 'power' . Default: 2 (powers of two)
        :param start: start position of a geometric progression in the case of ascending=False. Default: 32
        :param channel_growth_rate: the way of calculating the number of feature maps between 'exponential', 'proportion', 'linear', 'power' and 'constant'. Default: 'exponential'
        :return: output channels after each convolutional layer
        # noqa
        """
        _validate_channel_growth_rate_param(channel_growth_rate)

        if channel_growth_rate == "exponential":
            self.max_channels = self._initial_max_channels
            return [in_channels] + [int(start * ratio**i) for i in range(n_layers)]

        if channel_growth_rate == "proportion":
            range_start = in_channels
            range_stop = int((sum(in_size) * 0.5) // len(in_size) + in_channels)
            range_step = (range_stop - in_channels) // n_layers

            _validate_range_step(range_step, n_layers)

            channels = list(range(range_start, range_stop + 1, range_step))[: n_layers + 1]
            self.max_channels = range_stop
            return channels

        if channel_growth_rate == "linear":
            self.max_channels = self.min_channels + n_layers
            return [in_channels] + [in_channels + i + 1 for i in range(n_layers)]

        if channel_growth_rate == "constant":
            self.max_channels = constant + 1
            return [in_channels] + [constant for _ in range(n_layers)]

        if channel_growth_rate == "power":
            self.max_channels = self._initial_max_channels
            return [in_channels] + [int((in_channels + i) ** ratio) for i in range(1, n_layers + 1)]

    @staticmethod
    def _calc_out_transpose_channels(
        in_channels: int,
        out_channels: int,
        n_layers: int,
        ratio: float = 2.0,
        constant: int = 1,
        channel_growth_rate: str = "exponential",
    ) -> List[int]:
        """
        The function to calculate output channels after each transposed convolutional layer

        :param in_channels: number of channels in the first input tensor
        :param out_channels: number of channels in the last tensor
        :param n_layers: number of the transposed convolution layers in the encoder part
        :param ratio: multiplier for the geometric progression of increasing channels (feature maps). Used for 'channel_growth_rate' as 'exponential' or 'power' . Default: 2 (powers of two)
        :param channel_growth_rate: the way of calculating the number of feature maps between 'exponential', 'proportion', 'linear', 'power' and 'constant'. Default: 'exponential'
        :return: output channels after each transposed convolutional layer
        # noqa
        """
        _validate_channel_growth_rate_param(channel_growth_rate)

        if channel_growth_rate == "exponential":
            return [int(in_channels / ratio**i) for i in range(n_layers)] + [out_channels]

        if channel_growth_rate == "proportion":
            channels = list(range(out_channels, in_channels, (in_channels - out_channels) // n_layers))[::-1]
            channels = channels[:n_layers]
            channels[-1] = out_channels
            return [in_channels] + channels

        if channel_growth_rate == "linear":
            return [in_channels] + [in_channels - i for i in range(1, n_layers)] + [out_channels]

        if channel_growth_rate == "constant":
            return [in_channels] + [constant for _ in range(n_layers - 1)] + [out_channels]

        if channel_growth_rate == "power":
            return (
                [in_channels] + [int(math.pow((n_layers - i + 1), ratio)) for i in range(1, n_layers)] + [out_channels]
            )
