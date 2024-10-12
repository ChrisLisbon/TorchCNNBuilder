from math import prod
from typing import Optional, Sequence

import torch
import torch.nn as nn

from torchcnnbuilder._validation import _validate_latent_layers, _validate_latent_shape


class LatentSpaceModule(nn.Module):
    """
    nn.Module that transforms a tensor from one latent space shape to another using fully connected layers

      Attributes:
          input_shape (Sequence[int]): the shape of the input tensor before transformation.
          output_shape (Sequence[int]): the shape of the output tensor after transformation
          resize (nn.Sequential): the transformation through linear layers
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
        :param activation: whether to apply an activation function after each linear layer. Default: False
        :param activation_function: activation function. Default: nn.ReLU(inplace=True)
        # noqa
        """

        super().__init__()
        _validate_latent_shape(input_shape)
        _validate_latent_shape(output_shape)
        _validate_latent_layers(n_layers)

        self.input_shape = input_shape
        self.output_shape = output_shape

        input_features = prod(input_shape)
        output_features = prod(output_shape)

        if n_layers > 1:
            log_input = torch.log(torch.tensor(input_features, dtype=torch.int))
            log_output = torch.log(torch.tensor(output_features, dtype=torch.int))
            features = torch.exp(torch.linspace(log_input, log_output, steps=n_layers + 1)).tolist()
            features = list(map(int, features))
        else:
            features = [input_features, output_features]

        latent_layers = []
        for i in range(n_layers):
            latent_layers.append(nn.Linear(features[i], features[i + 1]))
            if activation_function is not None:
                latent_layers.append(activation_function)

        self.resize = nn.Sequential(*latent_layers)

    def forward(self, x):
        """
        Forward pass of the model

        :param x: tensor before forward pass
        :return: tensor after forward pass
        # noqa
        """
        return self.resize(x.view(-1)).view(self.output_shape)
