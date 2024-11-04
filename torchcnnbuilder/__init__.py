"""
**TorchCNNBuilder** is an open-source framework for the automatic creation of CNN architectures.
This framework should first of all help researchers in the applicability of CNN models for a huge range of tasks,
taking over most of the writing of the architecture code. This framework is distributed under the 3-Clause BSD license.
All the functionality is written only using `pytorch` *(no third-party dependencies)*.

.. include:: ../.docs/main.md
"""
from torchcnnbuilder._constants import (
    DEFAULT_CONV_PARAMS,
    DEFAULT_TRANSPOSE_CONV_PARAMS,
    MAX_PARAMS_NUMBER_PER_LAYER,
)
from torchcnnbuilder._formulas import (
    conv1d_out,
    conv2d_out,
    conv3d_out,
    conv_transpose1d_out,
    conv_transpose2d_out,
    conv_transpose3d_out,
)
from torchcnnbuilder._version import __version__
