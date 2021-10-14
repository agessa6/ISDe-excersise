from random import randint

import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def load_mnist_data():
    data = read_csv(
        "https://github.com/unica-isde/isde/raw/master/data/mnist_data.csv")
    data = np.array(data)  # casting DataFrame to ndarray
    x = data[:, 1:]
    y = data[:, 0]
    return x, y


def plot_ten_digits(x, y=None):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img = x[i, :].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        if y is not None:
            plt.title("Label: " + str(y[i]))
