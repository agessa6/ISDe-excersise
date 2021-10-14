import numpy as np
from utils import load_mnist_data, plot_ten_digits
from conv_1d_kernels import CConvKernelMovingAverage, CConvKernelTriangle, CConvKernelCombo
import matplotlib.pyplot as plt
from random import randint

combo_kernel = CConvKernelCombo()
moving_average_kernel = CConvKernelMovingAverage()
triangle_kernel = CConvKernelTriangle()


x, y = load_mnist_data()

labels = np.unique(y)
images = np.zeros(shape=(labels.size, x.shape[1]))
for i in range(labels.size):
    all_images_of_current_label = x[y == labels[i], :]
    random_index = randint(0, all_images_of_current_label.shape[0])
    images[i, :] = all_images_of_current_label[random_index, :]
plot_ten_digits(images, labels)
plt.suptitle("Without filter")
plt.savefig('./images/' + "Without filter" + "images.pdf")


#   applying moving average kernel
moving_average_images = np.array([moving_average_kernel.kernel(images[i])
                                  for i in range(labels.shape[0])])
plot_ten_digits(moving_average_images, labels)
plt.suptitle("Moving average filter")
plt.savefig('./images/' + "Moving average filter" + "images.pdf")


#   applying triangle kernel
triangle_images = np.array([triangle_kernel.kernel(images[i])
                            for i in range(labels.shape[0])])
plot_ten_digits(triangle_images, labels)
plt.suptitle("Triangle filter")
plt.savefig('./images/' + "Triangle filter" + "images.pdf")


#   applying combo kernel
combo_images = \
    np.array(
        [combo_kernel.combo(images[i], moving_average_kernel.mask, triangle_kernel.mask, moving_average_kernel.mask)
         for i in range(labels.shape[0])])
plot_ten_digits(combo_images, labels)
plt.suptitle("Combo filter")
plt.savefig('./images/' + "Combo filter" + "images.pdf")
