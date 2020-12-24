from ..param import ParameterMap
import torch
import copy


class ParameterMaps:
    intercepts: list = None
    decay: ParameterMap = None
    shape: tuple = None
    affine: torch.tensor = None

    def __len__(self):
        return len(self.intercepts) + 1

    def __iter__(self):
        maps = self.intercepts + [self.decay]
        for map in maps:
            yield map

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)
