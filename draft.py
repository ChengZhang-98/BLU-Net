import glob
import itertools
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.utils import Sequence
from keras import layers
from skimage import morphology as morph
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DownSamplingBlock(layers.Layer):
    def __init__(self, filters, conv2d_params, name="Contracting"):
        super(DownSamplingBlock, self).__init__(name=name)
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")
        self.conv1 = layers.Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_1")
        self.conv2 = layers.Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")


def f1(name):
    def f2(age):
        print(name, age)
    return f2


if __name__ == '__main__':
    name = list("Cheng")
    f2 = f1(name)
    f2(23)
    name[0]="CC"
    f2(22)
    print("done")
