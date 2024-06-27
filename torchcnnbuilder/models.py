from typing import Optional, Sequence, Union

import torch.nn as nn

from torchcnnbuilder.builder import Builder


# ------------------------------------
# CNN Forecaster pattern class
# ------------------------------------
class ForecasterBase(nn.Module):
    """
      The template class of the time series prediction CNN-architecture. The source of the original `article code
    <https://github.com/ITMO-NSS-team/ice-concentration-prediction-paper?ysclid=lrhxbvsk8s328492826>`_.

      Attributes:
          convolve (nn.Sequential): convolutional sequence - encoder part
          transpose (nn.Sequential): transpose convolutional sequence - decoder part
          conv_channels (List[int]): list of output channels after each convolutional layer
          transpose_conv_channels (List[int]): list of output channels after each transposed convolutional layer
          conv_layers (List[tuple]): list of output tensor sizes after each convolutional layer
          transpose_conv_layers (List[tuple]): list of output tensor sizes after each transposed convolutional layer
    # noqa
    """

    def __init__(
        self,
        input_size: Sequence[int],
        n_layers: int,
        in_channels: int,
        out_channels: int,
        conv_dim: int = 2,
        n_transpose_layers: Optional[int] = None,
        convolve_params: Optional[dict] = None,
        transpose_convolve_params: Optional[dict] = None,
        activation_function: nn.Module = nn.ReLU(inplace=True),
        finish_activation_function: Union[str, Optional[nn.Module]] = None,
        normalization: Optional[str] = None,
    ) -> None:
        """
        The constructor for ForecasterBase

        :param input_size: input size of the input tensor
        :param n_layers: number of the convolution layers in the encoder part
        :param in_channels: number of channels in the first input tensor (prehistory size)
        :param out_channels: number of channels in the last output tensor (forecasting size)
        :param conv_dim: the dimension of the convolutional operation 2 or 3. Default: 2
        :param n_transpose_layers: number of the transpose convolution layers in the encoder part. Default: None (same as n_layers)
        :param convolve_params: parameters of convolutional layers (by default same as in torch). Default: None
        :param transpose_convolve_params: parameters of transpose convolutional layers (by default same as in torch). Default: None
        :param activation_function: activation function. Default: nn.ReLU(inplace=True)
        :param finish_activation_function: last activation function, can be same as activation_function (str 'same'). Default: None
        :param normalization: choice of normalization between str 'dropout' and 'batchnorm'. Default: None
        # noqa
        """
        super(ForecasterBase, self).__init__()
        builder = Builder(
            input_size=input_size,
            activation_function=activation_function,
            finish_activation_function=finish_activation_function,
        )

        if n_transpose_layers is None:
            n_transpose_layers = n_layers

        if convolve_params is None:
            convolve_params = builder.default_convolve_params

        if transpose_convolve_params is None:
            transpose_convolve_params = builder.default_transpose_params

        channel_growth_rate = "linear" if conv_dim == 3 else "proportion"

        self.convolve = builder.build_convolve_sequence(
            n_layers=n_layers,
            in_channels=in_channels,
            params=convolve_params,
            conv_dim=conv_dim,
            normalization=normalization,
            channel_growth_rate=channel_growth_rate,
        )

        self.transpose = builder.build_transpose_convolve_sequence(
            n_layers=n_transpose_layers,
            in_channels=builder.conv_channels[-1],
            out_channels=out_channels,
            params=transpose_convolve_params,
            conv_dim=conv_dim,
            normalization=normalization,
            channel_growth_rate=channel_growth_rate,
        )

        self.conv_channels = builder.conv_channels
        self.transpose_conv_channels = builder.transpose_conv_channels
        self.conv_layers = builder.conv_layers
        self.transpose_conv_layers = builder.transpose_conv_layers

    def forward(self, x):
        """
        Forward pass of the model

        :param x: tensor before forward pass
        :return: tensor after forward pass
        # noqa
        """
        x = self.convolve(x)
        x = self.transpose(x)
        return x
