import numpy as np
from conv_1d_kernels import CConvKernel


class CConvKernelTriangle(CConvKernel):

    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)
        self._mask = self.kernel_mask()

    def kernel_mask(self):
        mask = np.zeros(self.kernel_size)

        for i in range(int(round(self.kernel_size/2))):
            mask[i] = i + 1
            mask[self.kernel_size - i - 1] = i + 1

        return np.multiply(mask,
                           1/pow(mask[int(self.kernel_size/2)],
                                 2))

