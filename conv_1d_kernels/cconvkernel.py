from abc import ABC
from copy import deepcopy

import numpy as np


class CConvKernel(ABC):

    def __init__(self, kernel_size=3):
        if kernel_size % 2 == 0:
            raise TypeError("The kernel size can't be an even number")

        self._kernel_size = kernel_size
        self._mask = None

    def kernel_mask(self):
        raise NotImplementedError("This method ins't implemented")

    def kernel(self, x, mask=None):
        if mask is None:
            mask = self._mask

        xp = deepcopy(x)
        half_kernel_size = int(self.kernel_size / 2)

        xp[half_kernel_size : xp.size - half_kernel_size] = \
            np.rint([np.dot(xp[(i - half_kernel_size) : (i + half_kernel_size + 1)],
                            mask)
                     for i in range(half_kernel_size,
                                    xp.size - half_kernel_size)]).astype(int)

        return xp

    @property
    def mask(self):
        return self._mask

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, new_size):
        self._kernel_size = new_size
        self._mask = self.kernel_mask()
