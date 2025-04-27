from typing import Dict

DEFAULT_CONV_PARAMS: Dict[str, int] = {
    "kernel_size": 3,
    "stride": 1,
    "padding": 0,
    "dilation": 1,
}
"""Default pytorch convolution params."""

DEFAULT_TRANSPOSE_CONV_PARAMS: Dict[str, int] = {
    "kernel_size": 3,
    "stride": 1,
    "padding": 0,
    "output_padding": 0,
    "dilation": 1,
}
"""Default pytorch transpose convolution params."""

_MAX_PARAMS_NUMBER_PER_LAYER: int = 500_000_000
