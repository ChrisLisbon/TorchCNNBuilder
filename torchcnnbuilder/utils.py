from typing import Type, Tuple, Dict

from torchcnnbuilder.builder import conv1d_out, conv2d_out, conv3d_out, conv_transpose1d_out, conv_transpose2d_out, conv_transpose3d_out
import torch.nn as nn


def _double_params(param: int) -> Tuple[int, int]:
    """
    Creating two parameters instead of one

    :param param: int param of some function
    :return Tuple[int, int]: doubled param
    """
    return param, param


def _triple_params(param: int) -> Tuple[int, int, int]:
    """
    Creating three parameters instead of one

    :param param: int param of some function
    :return Tuple[int, int, int]: tripled param
    """
    return param, param, param


def _set_conv_params(default_params: Dict[str, int], params: Dict[str, int]) -> Dict[str, int]:
    """
    Set convolution or transpose convolution params by using default one

    :param default_params: default convolution or transpose convolution params
    :param params: new users convolution params
    :return Dict[str, int]: set convolution params
    """

    default_params = default_params.copy()
    if params:
        for key, value in params.items():
            default_params[key] = value
    return default_params


def _select_conv_dimension(conv_dim: int,
                           transpose: bool = False) -> Type[nn.Module]:
    """
    The function to select nn.ConvNd

    :param conv_dim: the dimension of the convolutional operation
    :param transpose: choice of conv types between transposed and ordinary one. Default: False
    :return: nn.Module object
    """
    if conv_dim == 1:
        if transpose:
            return nn.ConvTranspose1d
        return nn.Conv1d
    elif conv_dim == 3:
        if transpose:
            return nn.ConvTranspose3d
        return nn.Conv3d
    if transpose:
        return nn.ConvTranspose2d

    # by default in all other cases
    return nn.Conv2d


def _select_conv_calc(conv_dim: int,
                      transpose: bool = False):
    """
    The function to select a way of calculating conv output

    :param conv_dim: the dimension of the convolutional operation
    :param transpose: choice of conv types between transposed and ordinary one. Default: False
    :return: one of functions to calculate conv or transposed conv output
    """
    if conv_dim == 1:
        if transpose:
            return conv_transpose1d_out
        return conv1d_out
    elif conv_dim == 3:
        if transpose:
            return conv_transpose3d_out
        return conv3d_out
    if transpose:
        return conv_transpose2d_out

    # by default in all other cases
    return conv2d_out


def _select_norm_dimension(conv_dim: int,
                           normalization: str = 'batchnorm') -> Type[nn.Module]:
    """
    The function to select nn.BatchNormNd or nn.DropoutNd

    :param conv_dim: the dimension of the convolutional operation
    :param normalization: choice of normalization between str 'dropout', 'batchnorm' and 'instancenorm'. Default: 'batchnorm'
    :return: nn.Module object
    """
    if normalization == 'dropout':
        if conv_dim == 1:
            return nn.Dropout1d
        elif conv_dim == 3:
            return nn.Dropout3d
        return nn.Dropout2d

    if normalization == 'batchnorm':
        if conv_dim == 1:
            return nn.BatchNorm1d
        elif conv_dim == 3:
            return nn.BatchNorm3d
        return nn.BatchNorm2d

    if normalization == 'instancenorm':
        if conv_dim == 1:
            return nn.InstanceNorm1d
        elif conv_dim == 3:
            return nn.InstanceNorm2d
        return nn.InstanceNorm2d

    # by default in all other cases
    return nn.BatchNorm2d


def _select_adaptive_pooling_dimension(conv_dim: int,
                                       pooling: str = 'avgpool') -> Type[nn.Module]:
    """
    The function to select nn.AdaptiveAvgPoolNd

    :param conv_dim: the dimension of the convolutional operation
    :param pooling: choice of adaptive pooling between str 'avgpool' and 'maxpool'. Default: 'avgpool'
    :return: nn.Module object
    """
    if pooling == 'avgpool':
        if conv_dim == 1:
            return nn.AdaptiveAvgPool1d
        elif conv_dim == 3:
            return nn.AdaptiveAvgPool3d
        return nn.AdaptiveAvgPool2d

    if pooling == 'maxpool':
        if conv_dim == 1:
            return nn.AdaptiveMaxPool1d
        elif conv_dim == 3:
            return nn.AdaptiveMaxPool3d
        return nn.AdaptiveMaxPool2d

    # by default in all other cases
    return nn.AdaptiveAvgPool2d
