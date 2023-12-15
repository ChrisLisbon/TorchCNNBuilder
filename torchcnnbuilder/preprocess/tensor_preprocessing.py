from typing import Sequence, Union, Any
from torch.utils.data import TensorDataset
import torch


def single_output_tensor(data: Sequence[Any],
                         forecast_len: int,
                         additional_x: Union[Sequence[Any], None] = None,
                         additional_is_array: Sequence[Any] = False,
                         additional_x_stack: bool = True,
                         threshold: Union[bool, float] = False,
                         x_binarize: bool = False) -> TensorDataset:
    """
    :param data: N-dimensional arrays, lists, numpy arrays, tensors etc.
    :param forecast_len: length of prediction for each y-train future tensor (target)
    :param additional_x: extra x-train data. Default: None
    :param additional_is_array: if additional x-train is a list/array of x_i data like other time series. Default: False
    :param additional_x_stack: if True stack each additional_x_i to x-train. Default: True
    :param threshold: binarization threshold for each y-tensor. Default: False
    :param x_binarize: binarization with threshold for each x-tensor. Default: False
    :return TensorDataset: TensorDataset of X-train and y-train
    """
    tensors = torch.Tensor(data)

    y_train_tensor = tensors[-forecast_len + 1:]
    x_train_tensor = tensors[:-forecast_len + 1]

    if additional_x is not None:
        additional_x = torch.Tensor(additional_x)

        if not additional_is_array:

            extra_x_train_tensor = additional_x[:-forecast_len + 1]
            if additional_x_stack:
                x_train_tensor = torch.stack([x_train_tensor, extra_x_train_tensor], dim=1)

        else:
            channels = [x_train_tensor]
            for array in additional_x:
                x_array = array[:-forecast_len + 1]
                channels.append(x_array)

            x_train_tensor = torch.stack(channels, dim=1)

    if not isinstance(threshold, bool):
        y_train_tensor[y_train_tensor > threshold] = 1
        y_train_tensor[y_train_tensor <= threshold] = 0

        if x_binarize:
            x_train_tensor[x_train_tensor > threshold] = 1
            x_train_tensor[x_train_tensor <= threshold] = 0

            if additional_x is not None and not additional_is_array and not additional_x_stack:
                extra_x_train_tensor[extra_x_train_tensor > threshold] = 1
                extra_x_train_tensor[extra_x_train_tensor <= threshold] = 0

    if additional_x is not None and not additional_is_array and not additional_x_stack:
        return TensorDataset(x_train_tensor[None], extra_x_train_tensor[None], y_train_tensor[None])

    return TensorDataset(x_train_tensor[None], y_train_tensor[None])


def multi_output_tensor(data: Sequence[Any],
                        forecast_len: int,
                        pre_history_len: int,
                        additional_x: Union[Sequence[Any], None] = None,
                        additional_is_array: Sequence[Any] = False,
                        additional_x_stack: bool = True,
                        threshold: Union[bool, float] = False,
                        x_binarize: bool = False) -> TensorDataset:
    """
   :param data: N-dimensional arrays, lists, numpy arrays, tensors etc.
   :param forecast_len: length of prediction for each y-train future tensor (target)
   :param pre_history_len: length of pre-history for each x-train future tensor
   :param additional_x: extra x-train data. Default: None
   :param additional_is_array: if additional x-train is a list/array of x_i data like other time series. Default: False
   :param additional_x_stack: if True stack each additional_x_i to x-train. Default: True
   :param threshold: binarization threshold for each y-tensor. Default: False
   :param x_binarize: binarization with threshold for each x-tensor. Default: False
   :return TensorDataset: TensorDataset of X-train and y-train
   """

    tensors = torch.Tensor(data)

    x_train_list, y_train_list = [], []

    if additional_x is not None:
        if additional_is_array:
            additional_x = torch.stack(list(map(torch.Tensor, additional_x)))
        else:
            additional_x = torch.Tensor(additional_x)
        extra_x_train_list = []

    for i in range(tensors.shape[0] - forecast_len - pre_history_len):
        x = tensors[i:i + pre_history_len]
        y = tensors[i + pre_history_len:i + pre_history_len + forecast_len]

        if additional_x is not None:

            if not additional_is_array:

                extra_x = additional_x[i:i + pre_history_len]
                if additional_x_stack:
                    x = torch.stack([x, extra_x], dim=1)
                else:
                    extra_x_train_list.append(extra_x)

            else:
                channels = [x]
                for array in additional_x:
                    x_array = array[i:i + pre_history_len]
                    channels.append(x_array)
                x = torch.stack(channels, dim=1)

        if not isinstance(threshold, bool):
            y[y > threshold] = 1
            y[y <= threshold] = 0

            if x_binarize:
                x[x > threshold] = 1
                x[x <= threshold] = 0

                if additional_x is not None and not additional_is_array and not additional_x_stack:
                    extra_x[extra_x > threshold] = 1
                    extra_x[extra_x <= threshold] = 0

        x_train_list.append(x)
        y_train_list.append(y)

    x_train_tensor, y_train_tensor = torch.stack(x_train_list), torch.stack(y_train_list)

    if additional_x is not None and not additional_is_array and not additional_x_stack:
        extra_x_train_tensor = torch.stack(extra_x_train_list)
        return TensorDataset(x_train_tensor, extra_x_train_tensor, y_train_tensor)

    return TensorDataset(x_train_tensor, y_train_tensor)