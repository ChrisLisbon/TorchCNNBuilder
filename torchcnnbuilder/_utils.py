from typing import Dict, Tuple, Type

import torch.nn as nn

from torchcnnbuilder._validation import (
    _validate_conv_dim,
    _validate_normalization_param,
    _validate_pooling_param,
)


def _double_params(param: int) -> Tuple[int, int]:
    """
    Creating two parameters instead of one

    :param param: int param of some function
    :return Tuple[int, int]: doubled param
    # noqa
    """
    return param, param


def _triple_params(param: int) -> Tuple[int, int, int]:
    """
    Creating three parameters instead of one

    :param param: int param of some function
    :return Tuple[int, int, int]: tripled param
    # noqa
    """
    return param, param, param


def _set_conv_params(default_params: Dict[str, int], params: Dict[str, int]) -> Dict[str, int]:
    """
    Set convolution or transpose convolution params by using default one

    :param default_params: default convolution or transpose convolution params
    :param params: new users convolution params
    :return Dict[str, int]: set convolution params
    # noqa
    """
    default_params = default_params.copy()
    if params:
        for key, value in params.items():
            default_params[key] = value
    return default_params


def _select_conv_dimension(conv_dim: int, transpose: bool = False) -> Type[nn.Module]:
    """
    The function to select nn.ConvNd

    :param conv_dim: the dimension of the convolutional operation
    :param transpose: choice of conv types between transposed and ordinary one. Default: False
    :return: nn.Module object
    # noqa
    """
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
    """
    The function to select nn.BatchNormNd or nn.DropoutNd

    :param conv_dim: the dimension of the convolutional operation
    :param normalization: choice of normalization between str 'dropout', 'batchnorm' and 'instancenorm'. Default: 'batchnorm'
    :return: nn.Module object
    # noqa
    """
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
    """
    The function to select nn.AdaptiveAvgPoolNd

    :param conv_dim: the dimension of the convolutional operation
    :param pooling: choice of adaptive pooling between str 'avgpool' and 'maxpool'. Default: 'avgpool'
    :return: nn.Module object
    # noqa
    """
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
