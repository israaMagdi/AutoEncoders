# Import libraries
from tensorflow.keras import datasets
import numpy as np


# Preprocessing
def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0  # normalize
    imgs = np.pad(
        imgs, ((0, 0), (2, 2), (3, 3)), constant_values=0.0
    )  # padding to save the information on corners
    imgs = np.expand_dims(
        imgs, -1
    )  # Because tensorflow expect the dimension of channels
    return imgs


# Load the fashion MNIST dataset
def load_dataset():
    """
    This function return the dataset as x_train, y_train, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test


# Apply preprocessing on images
# x_train = preprocess(x_train)
# x_test = preprocess(x_test)
