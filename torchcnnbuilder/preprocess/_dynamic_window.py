from typing import Any, Sequence, Union

import torch
from torch.utils.data import TensorDataset


def single_output_tensor(
    data: Sequence[Any],
    forecast_len: int,
    additional_x: Union[Sequence[Any], None] = None,
    additional_is_array: bool = False,
    additional_x_stack: bool = True,
    threshold: Union[bool, float] = False,
    x_binarize: bool = False,
) -> TensorDataset:
    """
    Preprocesses a time series into a tensor with input (X) and output (Y) parts for single-step predictions.
    See the tensor transformation diagram below:

    ![image](../../.docs/media/single_output_tensor.png)

    This function prepares input and output tensors for training a model with a given forecast length. Additional
    optional data can be provided to expand the input features with extra time series.

    Args:
        data (Sequence[Any]): The time series data in an N-dimensional format (e.g., list, numpy array, or tensor).
        forecast_len (int): Number of time steps for each target output tensor.
        additional_x (Union[Sequence[Any], None], optional): Extra data to add as additional input features.
            Defaults to None.
        additional_is_array (bool, optional): If True, treats `additional_x` as a collection of separate
            time series. Defaults to False.
        additional_x_stack (bool, optional): If True, stacks `additional_x` features onto input data (X).
            If False and `additional_is_array` is also False, `additional_x` will be returned separately.
            Defaults to True.
        threshold (Union[bool, float], optional): Threshold for binarizing output tensor (Y). If set to a float,
            values above the threshold are set to 1, and values below are set to 0. Defaults to False.
        x_binarize (bool, optional): If True, applies the same binarization as `threshold` to the input tensor (X).
            Defaults to False.

    Returns:
       A dataset containing the input and output tensors for training.

    Raises:
        ValueError: If `forecast_len` is greater than the data length.
        TypeError: If the types of `data` or `additional_x` are unsupported.
    """
    tensors = torch.Tensor(data)

    y_train_tensor = tensors[-forecast_len:]
    x_train_tensor = tensors[:-forecast_len]

    if additional_x is not None:
        additional_x = torch.Tensor(additional_x)

        if not additional_is_array:
            extra_x_train_tensor = additional_x[:-forecast_len]
            if additional_x_stack:
                x_train_tensor = torch.stack([x_train_tensor, extra_x_train_tensor], dim=1)

        else:
            channels = [x_train_tensor]
            for array in additional_x:
                x_array = array[:-forecast_len]
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


def multi_output_tensor(
    data: Sequence[Any],
    forecast_len: int,
    pre_history_len: int,
    additional_x: Union[Sequence[Any], None] = None,
    additional_is_array: bool = False,
    additional_x_stack: bool = True,
    threshold: Union[bool, float] = False,
    x_binarize: bool = False,
) -> TensorDataset:
    """
    Preprocesses a time series into a sliding-window tensor with input (X) and output (Y)
    parts for multi-step predictions. See the tensor transformation diagram below:

    ![image](../../.docs/media/single_output_tensor.png)

    This function prepares input and output tensors for training a model with a given forecast length. Additional
    optional data can be provided to expand the input features with extra time series.

    Args:
        data (Sequence[Any]): Time series data in an N-dimensional format (e.g., list, numpy array, or tensor).
        forecast_len (int): Number of time steps for each target output tensor.
        pre_history_len (int): Length of the time window for input tensors.
        additional_x (Union[Sequence[Any], None], optional): Additional input data to augment features.
            Defaults to None.
        additional_is_array (bool, optional): If True, treats `additional_x` as separate time series. Defaults to False.
        additional_x_stack (bool, optional): If True, stacks `additional_x` features onto input data (X). If False
            and `additional_is_array` is also False, `additional_x` is returned separately. Defaults to True.
        threshold (Union[bool, float], optional): Threshold for binarizing the output tensor (Y). If set to a float,
            values above the threshold are set to 1, and values below are set to 0. Defaults to False.
        x_binarize (bool, optional): If True, applies binarization to the input tensor (X) as per `threshold`.
            Defaults to False.

    Returns:
        A dataset containing input and output tensors for training.

    Raises:
        ValueError: If `forecast_len` or `pre_history_len` exceed the data length.
        TypeError: If the types of `data` or `additional_x` are unsupported.
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
        x = tensors[i : i + pre_history_len]
        y = tensors[i + pre_history_len : i + pre_history_len + forecast_len]

        if additional_x is not None:
            if not additional_is_array:
                extra_x = additional_x[i : i + pre_history_len]
                if additional_x_stack:
                    x = torch.stack([x, extra_x], dim=1)
                else:
                    extra_x_train_list.append(extra_x)

            else:
                channels = [x]
                for array in additional_x:
                    x_array = array[i : i + pre_history_len]
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
