from typing import Sequence, Tuple, Optional, List
from torch import tensor


def _validate_difference_in_dimensions(input_size: Sequence[int], conv_dim: int) -> None:
    if len(input_size) - conv_dim not in (0, 1):
        raise ValueError(
            f'The difference in dimensions between input_size {input_size=} '
            f'and convolution {conv_dim=} should not be more than 1 '
            f'(input_size.shape - conv_dim should be equal to 1 or 0)')


def _validate_available_layers(layer: int, input_layer_size: Tuple[int, ...] , minimum_feature_map_size: Tuple[int, ...]) -> None:
    if all(tensor(input_layer_size) < tensor(minimum_feature_map_size)[:len(input_layer_size)]):
        raise ValueError(f'Input size and parameters can not provide more than {layer + 1} layers')


def _validate_max_channels_number(input_channels_count_list, layer, max_channels) -> None:
    if input_channels_count_list[layer] > max_channels:
        raise ValueError(f'There is too many channels. Max channels {max_channels} [layer {layer}]')


def _validate_min_channels_number(input_channels_count_list, layer, transpose: bool = False):
    if transpose:
        if input_channels_count_list[layer] < 1:
            raise ValueError(f'There is too few channels. You can not provide less then 1 channel [layer {layer}]')


'''
def _validate_min_channels_number()

elif input_channels_count_list[layer] < self.min_channels and layer != 0 and not ascending:
    raise ValueError(f'There is too few channels. Min channels {self.min_channels} [layer {layer}]')

'''


def _validate_build_transpose_convolve_init(in_channels: Optional[int], conv_channels: List[Tuple[int, ...], ...]) -> None:
    if in_channels is None and not conv_channels:
        raise ValueError(f'You should specify in_channels or use build_convolve_sequence before transposed one')