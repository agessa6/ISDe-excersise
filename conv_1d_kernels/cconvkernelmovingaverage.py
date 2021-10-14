from conv_1d_kernels import CConvKernel


class CConvKernelMovingAverage(CConvKernel):

    def __init__(self, kernel_size=3):
        super().__init__(kernel_size)
        self._mask = self.kernel_mask()

    def kernel_mask(self):
        return [1 / self.kernel_size for i in range(self.kernel_size)]

