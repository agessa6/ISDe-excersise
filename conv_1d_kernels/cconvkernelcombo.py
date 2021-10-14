from conv_1d_kernels import CConvKernel

class CConvKernelCombo(CConvKernel):

    def kernel_mask(self):
        super().kernel_mask()

    def combo(self, x, *kernel_masks):
        xp_copy = x.copy()

        for mask in kernel_masks:
            xp = self.kernel(xp_copy,
                             mask)
            xp_copy = xp.copy()

        return xp