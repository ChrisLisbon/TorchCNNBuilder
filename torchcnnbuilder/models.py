from typing import Union, Sequence, Optional

from torchcnnbuilder.builder import EncoderBuilder
import torch.nn as nn


# ------------------------------------
# CNN Forecaster pattern class
# ------------------------------------
class ForecasterBase(nn.Module):
    def __int__(self,
                input_size: Sequence[int],
                n_layers: int,
                in_channels: int,
                out_channels: int,
                n_transpose_layers: Optional[int] = None,
                convolve_params: Optional[dict] = None,
                transpose_convolve_params: Optional[dict] = None,
                activation_function: nn.Module = nn.ReLU(inplace=True),
                finish_activation_function: Union[str, Optional[nn.Module]] = None,
                normalization: Optional[str] = None):

        builder = EncoderBuilder(input_size=input_size,
                                 activation_function=activation_function,
                                 finish_activation_function=finish_activation_function)

        if n_transpose_layers is None:
            n_transpose_layers = n_layers

        if convolve_params is None:
            convolve_params = builder.default_convolve_params

        if transpose_convolve_params is None:
            transpose_convolve_params = builder.default_transpose_params

        self.convolve = builder.build_convolve_sequence(n_layers=n_layers,
                                                        in_channels=in_channels,
                                                        params=convolve_params,
                                                        normalization=normalization,
                                                        ascending=True)

        self.transpose = builder.build_transpose_convolve_sequence(n_layers=n_transpose_layers,
                                                                   in_channels=builder.conv_channels[-1],
                                                                   out_channels=out_channels,
                                                                   params=transpose_convolve_params,
                                                                   normalization=normalization,
                                                                   ascending=True)

    def forward(self, x):
        x = self.convolve(x)
        x = self.transpose(x)
        return x
