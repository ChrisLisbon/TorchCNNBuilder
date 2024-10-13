from math import prod
from typing import Optional, Sequence

import torch
import torch.nn as nn

from torchcnnbuilder._validation import (
    _validate_latent_layers,
    _validate_latent_shape,
    _validate_warning_huge_linear_weights_matrix,
)


# ------------------------------------
# Linear LatentSpaceModule class
# ------------------------------------
class LatentSpaceModule(nn.Module):
    """
    nn.Module that transforms a tensor from one latent space shape to another using linear fully connected layers

      Attributes:
          input_shape (Sequence[int]): the shape of the input tensor before transformation.
          output_shape (Sequence[int]): the shape of the output tensor after transformation
    # noqa
    """

    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        n_layers: int = 1,
        activation_function: Optional[nn.Module] = None,
    ):
        """
        The constructor for LatentSpaceModule

        :param input_shape: the shape of the input tensor
        :param output_shape: the desired shape of the output tensor
        :param n_layers: number of linear layers to use in the transformation. Default: 1
        :param activation_function: activation function. Default: None
        # noqa
        """

        super().__init__()
        _validate_latent_shape(input_shape)
        _validate_latent_shape(output_shape)
        _validate_latent_layers(n_layers)

        self._input_shape = input_shape
        self._output_shape = output_shape
        self._n_layers = n_layers
        self._activation_function = activation_function

        input_features = prod(self._input_shape)
        output_features = prod(self._output_shape)

        flatten_layer = nn.Flatten()
        unflatten_layer = nn.Unflatten(1, self._output_shape)

        if n_layers > 1:
            log_input = torch.log(torch.tensor(input_features, dtype=torch.int))
            log_output = torch.log(torch.tensor(output_features, dtype=torch.int))
            features = torch.exp(torch.linspace(log_input, log_output, steps=n_layers + 1)).tolist()
            features = list(map(int, features))
        else:
            features = [input_features, output_features]

        latent_layers = [flatten_layer]
        for i in range(self._n_layers):
            in_features, out_features = features[i], features[i + 1]

            _validate_warning_huge_linear_weights_matrix(
                in_features, out_features, level=f"linear latent layer number {i}"
            )

            latent_layers.append(nn.Linear(in_features, out_features))
            if activation_function is not None:
                latent_layers.append(activation_function)

        latent_layers.append(unflatten_layer)
        self._resize = nn.Sequential(*latent_layers)

    @property
    def input_shape(self) -> Sequence[int]:
        return self._input_shape

    @property
    def output_shape(self) -> Sequence[int]:
        return self._output_shape

    def forward(self, x):
        """
        Forward pass of the model

        :param x: tensor before forward pass
        :return: tensor after forward pass
        # noqa
        """
        return self._resize(x)

    def __repr__(self):
        """
        Custom string representation of the module
        """
        default_repr_model_params = [f"input_shape={self._input_shape}", f"output_shape={self._output_shape}"]

        if self._n_layers != 1:
            default_repr_model_params.append(f"n_layers={self._n_layers}")

        if self._activation_function is not None:
            default_repr_model_params.append(f"activation_function={self._activation_function}")

        return f"LatentSpaceModule({', '.join(default_repr_model_params)})"
