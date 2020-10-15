# -*- coding: utf-8 -*-
"""Convolution layers."""

import torch
from torch import nn as tnn
from ..core.pyutils import make_list, rep_list, getargs
from copy import copy
import math
import inspect

# NOTE:
# My version of Conv allows parameters to be overridden at eval time.
# This probably clashes with these parameters being declared as __constants__
# in torch.nn._ConvND. I think this __constants__ mechanics is only used
# In TorchScript, so it's fine for now.
#
# After some googling, I think it is alright as long as the *attribute*
# is not mutated...
# Some references:
# .. https://pytorch.org/docs/stable/jit.html#frequently-asked-questions
# .. https://discuss.pytorch.org/t/why-do-we-use-constants-or-final/70331/2
#
# Note that optional submodules can also be added to __constants__ in a
# hacky way:
# https://discuss.pytorch.org/t/why-do-we-use-constants-or-final/70331/4

# I should probably move this somewhere else
_activations = {
    # linear units
    'relu': tnn.ReLU,               # clip (, 0]
    'relu6': tnn.ReLU6,             # clip (, 0] and [6, )
    'leaky_relu': tnn.LeakyReLU,    # mult factor for negative slope
    'prelu': tnn.PReLU,             # LeakyReLU with learnable factor
    'rrelu': tnn.RReLU,             # LeakyReLU with random factor
    # sigmoid / softmax / soft functions
    'sigmoid': tnn.Sigmoid,
    'logsigmoid': tnn.LogSigmoid,    # log(sigmod)
    'hardsigmoid': tnn.Hardsigmoid,  # linear approximation of sigmoid
    'softmax': tnn.Softmax,          # multivariate sigmoid
    'logsoftmax': tnn.LogSoftmax,    # log(softmax)
    # smooth approximations
    'hardswish': tnn.Hardswish,      # 'smooth' RELU (quadratic)
    'softplus': tnn.Softplus,        # 'smooth' ReLU (logsumexp)
    'gelu': tnn.GELU,                # 'smooth' RELU (Gaussian cdf)
    'elu': tnn.ELU,                  # 'smooth' ReLU (exp-1)
    'selu': tnn.SELU,                #               (scaled ELU)
    'celu': tnn.CELU,                #               (~= SELU)
    'softsign': tnn.Softsign,        # 'smooth' sign function
    # shrinkage
    'softshrink': tnn.Softshrink,    # soft-thresholding (subtract constant)
    'hardshrink': tnn.Hardshrink,    # clip [-lam, +lam]
    # ranh
    'tanh': tnn.Tanh,                # hyperbolic tangent
    'hardtanh': tnn.Hardtanh,        # linear approximation of tanh
    'tanhshrink': tnn.Tanhshrink,    # shrink by tanh
}


class Conv(tnn.Module):
    """Convolution layer (with activation).

    Applies a convolution over an input signal.
    Optionally: apply an activation function to the output.

    """
    def __init__(self, dim, *args, **kwargs):
        """
        Args:
            dim (int): Dimension of the convolving kernel (1|2|3)
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution.
                Default: 1
            padding (int or tuple, optional): Zero-padding added to all
                three sides of the input. Default: 0
            padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to
                the output. Default: ``True``
            transposed (bool, optional): Transposed convolution.
            `Default: ``True``
            activation (type or function, optional): Constructor of an
                activation function. Default: ``None``
            try_inplace (bool, optional): Apply activation inplace if
                possible (i.e., not (is_leaf and requires_grad).
                Default: True.

        """
        super().__init__()

        # Get additional arguments that are not present in torch's conv
        transposed, activation, try_inplace = getargs(
            [('transposed', 10, False),
             ('activation', 11, None),
             ('try_inplace', 12, True)],
            args, kwargs, consume=True)

        # Store dimension
        self.dim = dim
        self.try_inplace = try_inplace

        # Select Conv
        if transposed:
            if dim == 1:
                self.conv = tnn.ConvTranspose1d(*args, **kwargs)
            elif dim == 2:
                self.conv = tnn.ConvTranspose2d(*args, **kwargs)
            elif dim == 3:
                self.conv = tnn.ConvTranspose3d(*args, **kwargs)
            else:
                NotImplementedError('Conv is only implemented in 1, 2, or 3D.')
        else:
            if dim == 1:
                self.conv = tnn.Conv1d(*args, **kwargs)
            elif dim == 2:
                self.conv = tnn.Conv2d(*args, **kwargs)
            elif dim == 3:
                self.conv = tnn.Conv3d(*args, **kwargs)
            else:
                NotImplementedError('Conv is only implemented in 1, 2, or 3D.')

        # Add activation
        #   an activation can be a class (typically a Module), which is
        #   then instantiated, or a callable (an already instantiated
        #   class or a more simple function).
        #   it is useful to accept both these cases as they allow to either:
        #       * have a learnable activation specific to this module
        #       * have a learnable activation shared with other modules
        #       * have a non-learnable activation
        if isinstance(activation, str):
            activation = _activations.get(activation.lower(), None)
        self.activation = activation() if inspect.isclass(activation) \
                          else activation if callable(activation) \
                          else None

    def forward(self, x, stride=None, padding=None, output_padding=None,
                dilation=None, padding_mode=None, activation=None):
        """Forward pass. Possibility to override constructed parameters.

        Args:
            x (torch.tensor): Input tensor
            stride (optional): Default: as constructed
            padding (optional): Default: as constructed
            output_padding (optional): Default: as constructed
            dilation (optional): Default: as constructed
            padding_mode (optional): Default: as constructed
            activation (optional): Default: as constructed

        Returns:
            x (torch.tensor): Convolved tensor

        """

        # Override constructed parameters
        conv = copy(self.conv)
        if stride is not None:
            conv.stride = make_list(stride, self.dim)
        if padding is not None:
            conv.padding = make_list(padding, self.dim)
            conv._padding_repeated_twice = rep_list(conv.padding, 2,
                                                    interleaved=True)
        if output_padding is not None:
            conv.output_padding = make_list(output_padding, self.dim)
        if dilation is not None:
            conv.dilation = make_list(dilation, self.dim)
        if padding_mode is not None:
            conv.padding_mode = padding_mode

        # Activation
        if activation is None:
            activation = copy(self.activation)
        else:
            activation = activation()
        if self.try_inplace \
                and hasattr(activation, 'inplace') \
                and not (x.is_leaf and x.requires_grad):
            activation.inplace = True

        # Convolution + Activation
        x = conv(x)
        if activation is not None:
            x = activation(x)
        return x

    def shape(self, x, stride=None, padding=None, output_padding=None,
              dilation=None, padding_mode=None, activation=None):
        """Compute output shape of the equivalent ``forward`` call.

        Args:
            x (torch.tensor): Input tensor
            stride (optional): Default: as constructed
            padding (optional): Default: as constructed
            output_padding (optional): Default: as constructed
            dilation (optional): Default: as constructed
            padding_mode (optional): Default: as constructed
            activation (optional): Default: as constructed

        Returns:
            x (tuple): Outptu shape

        """

        # Override constructed parameters
        conv = copy(self.conv)
        if stride is not None:
            conv.stride = make_list(stride, self.dim)
        if padding is not None:
            conv.padding = make_list(padding, self.dim)
            conv._padding_repeated_twice = rep_list(conv.padding, 2,
                                                   interleaved=True)
        if output_padding is not None:
            conv.output_padding = make_list(output_padding, self.dim)
        if dilation is not None:
            conv.dilation = make_list(dilation, self.dim)
        if padding_mode is not None:
            conv.padding_mode = padding_mode

        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation
        kernel_size = conv.kernel_size
        output_padding = conv.output_padding
        transposed = conv.transposed

        N = x.shape[0]
        C = self.conv.out_channels
        shape = [N, C]
        for i, inp in enumerate(x.shape[2:]):
            if transposed:
                shape.append(
                    (inp-1)*stride[i]-2*padding[i]
                    +dilation[i]*(kernel_size[i]-1)
                     +output_padding[i]+1
                )
            else:
                shape.append(math.floor(
                    (inp+2*padding[i]-dilation[i]*(kernel_size[i]-1)-1)
                    /stride[i] + 1
                ))
        return tuple(shape)

