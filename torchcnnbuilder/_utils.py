from typing import Dict, Optional, Tuple, Type

import torch.nn as nn

from torchcnnbuilder._validation import (
    _validate_conv_dim,
    _validate_normalization_param,
    _validate_pooling_param,
)


def _double_params(param: int) -> Tuple[int, int]:
    return param, param


def _triple_params(param: int) -> Tuple[int, int, int]:
    return param, param, param


def _set_conv_params(default_params: Dict[str, int], params: Optional[Dict[str, int]]) -> Dict[str, int]:
    default_params = default_params.copy()
    if params is not None:
        for key, value in params.items():
            default_params[key] = value
    return default_params


def _select_conv_dimension(conv_dim: int, transpose: bool = False) -> Type[nn.Module]:
    _validate_conv_dim(conv_dim)

    if conv_dim == 1:
        if transpose:
            return nn.ConvTranspose1d
        return nn.Conv1d

    if conv_dim == 2:
        if transpose:
            return nn.ConvTranspose2d
        return nn.Conv2d

    if conv_dim == 3:
        if transpose:
            return nn.ConvTranspose3d
        return nn.Conv3d


def _select_norm_dimension(conv_dim: int, normalization: str = "batchnorm") -> Type[nn.Module]:
    _validate_normalization_param(normalization)

    if normalization == "dropout":
        if conv_dim == 1:
            return nn.Dropout1d
        if conv_dim == 2:
            return nn.Dropout2d
        if conv_dim == 3:
            return nn.Dropout3d

    if normalization == "batchnorm":
        if conv_dim == 1:
            return nn.BatchNorm1d
        if conv_dim == 2:
            return nn.BatchNorm2d
        if conv_dim == 3:
            return nn.BatchNorm3d

    if normalization == "instancenorm":
        if conv_dim == 1:
            return nn.InstanceNorm1d
        if conv_dim == 2:
            return nn.InstanceNorm2d
        if conv_dim == 3:
            return nn.InstanceNorm2d


def _select_adaptive_pooling_dimension(conv_dim: int, pooling: str = "avgpool") -> Type[nn.Module]:
    _validate_pooling_param(pooling)
    _validate_conv_dim(conv_dim)

    if pooling == "avgpool":
        if conv_dim == 1:
            return nn.AdaptiveAvgPool1d
        if conv_dim == 2:
            return nn.AdaptiveAvgPool2d
        if conv_dim == 3:
            return nn.AdaptiveAvgPool3d

    if pooling == "maxpool":
        if conv_dim == 1:
            return nn.AdaptiveMaxPool1d
        if conv_dim == 2:
            return nn.AdaptiveMaxPool2d
        elif conv_dim == 3:
            return nn.AdaptiveMaxPool3d
