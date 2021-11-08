# WIP: add sliding window inferer and maybe some gradCAM / SEAM stuff at a later point

import math
import torch
from nitorch.core import utils
from nitorch.nn.base import Module


class SlidingWindow(Module):
    """
    Sliding window (patch-wise) inference.
    """
    def __init__(self,
                 patch_size,
                 overlap=0,
                 reduction='mean'):
        """
        
        """
        super().__init__()
        self.patch_size = patch_size
        self.reduction = reduction
        if overlap < 0.5 and overlap > 0:
            if isinstance(patch_size, list):
                if isinstance(overlap, list):
                    self.stride = [(1-overlap[i])*p for i, p in enumerate(patch_size)]
                else:
                    self.stride = [(1-overlap)*p for p in patch_size]
            else:
                if isinstance(overlap, list):
                    self.stride = [(1-o)*patch_size for o in overlap]
                else:
                    self.stride = (1-overlap)*patch_size
        else:
            self.stride = None

    def forward(self, x, model):
        shape = x.shape[2:]
        dim = len(shape)
        x = utils.unfold(x, kernel_size=self.patch_size, stride=self.stride, collapse=True)
        x = torch.chunk(x, 1, dim=2)
        x = [x_.reshape(tuple(x_.shape[:2])+tuple(x_.shape[3:])) for x_ in x]
        x = [model(x_) for x_ in x]
        x = [x_.unsqueeze(dim=2) for x_ in x]
        x = torch.cat(x, dim=2)
        x = utils.fold(x, dim=dim, stride=self.stride, collapsed=True, shape=shape, reduction=self.reduction)
        return x