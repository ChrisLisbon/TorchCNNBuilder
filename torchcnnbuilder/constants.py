from typing import Dict

DEFAULT_CONV_PARAMS: Dict[str, int] = {
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1
}

DEFAULT_TRANSPOSE_CONV_PARAMS: Dict[str, int] = {
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1
}