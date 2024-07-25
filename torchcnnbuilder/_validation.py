from typing import List, Optional, Sequence, Tuple

from torch import tensor


def _validate_difference_in_dimensions(input_size: Sequence[int], conv_dim: int) -> None:
    if len(input_size) - conv_dim not in (0, 1):
        raise ValueError(
            f"The difference in dimensions between input_size {input_size=} "
            f"and convolution {conv_dim=} should not be more than 1 "
            f"(input_size.shape - conv_dim should be equal to 1 or 0)"
        )


def _validate_available_layers(
    layer: int,
    input_layer_size: Tuple[int],
    minimum_feature_map_size: Sequence[int] | int,
) -> None:
    if all(tensor(input_layer_size) < tensor(minimum_feature_map_size)[: len(input_layer_size)]):
        raise ValueError(f"Input size and parameters can not provide more than {layer + 1} layers")


def _validate_max_channels_number(layer: int, input_channels_count_list: List[int], max_channels: int) -> None:
    if input_channels_count_list[layer] > max_channels:
        raise ValueError(
            f"There is too many channels [[{input_channels_count_list[layer]}]]. "
            f"Max channels {max_channels} [layer {layer}]"
        )


def _validate_min_channels_number(layer: int, input_channels_count_list: List[int], min_channels: int) -> None:
    if input_channels_count_list[layer] < min_channels:
        raise ValueError(
            f"There is too few channels [{input_channels_count_list[layer]}]. "
            f"You can not provide less then 1 channel [layer {layer}]"
        )


def _validate_build_transpose_convolve_init(in_channels: Optional[int], conv_channels: List[Tuple[int, ...]]) -> None:
    if in_channels is None and not conv_channels:
        raise ValueError("You should specify in_channels or use build_convolve_sequence before transposed one")


def _validate_range_step(range_step: int, n_layers: int) -> None:
    if range_step == 0:
        raise ValueError(f"Input size and parameters can not provide {n_layers} layers, try other model parameters")


def _validate_channel_growth_rate_param(channel_growth_rate: str) -> None:
    available_growth_rate = ("exponential", "proportion", "linear", "power", "constant")
    if channel_growth_rate not in available_growth_rate:
        raise ValueError(f"There is no param {channel_growth_rate=}. Choose something from {available_growth_rate=}")


def _validate_normalization_param(normalization: str) -> None:
    available_normalization = ("batchnorm", "instancenorm", "dropout")
    if normalization not in available_normalization:
        raise ValueError(f"There is no param {normalization=}. Choose something from {available_normalization=}")


def _validate_pooling_param(pooling: str) -> None:
    available_pooling = ("avgpool", "maxpool")
    if pooling not in available_pooling:
        raise ValueError(f"There is no param {pooling=}. Choose something from {available_pooling=}")


def _validate_conv_dim(conv_dim: int) -> None:
    if not 1 <= conv_dim <= 3:
        raise ValueError(f"You should choose convolution dimension between 1 and 3. You have {conv_dim}=")


def _validate_sequence_length(sequence: Sequence[int], length: int) -> None:
    input_length = len(sequence)
    if input_length > length:
        raise ValueError(f"{input_length=} should be less than {length=}")


def _validate_input_size_is_not_none(input_size: Optional[Sequence[int]]) -> None:
    if input_size is None:
        raise ValueError("You need to specify builder input_size in order to use this method")
