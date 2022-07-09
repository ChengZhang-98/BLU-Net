import glob
import itertools
import operator
import os
import time
import copy
from typing import List, Tuple

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.utils import Sequence
from keras import layers
from skimage import morphology as morph
from skimage import transform
from tqdm import tqdm
from scipy import interpolate
import tensorflow_probability as tfp
import elasticdeform
from matplotlib import gridspec
import tensorflow_addons as tfa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, )
        print('Elapsed: %s' % (time.time() - self.tstart))


if __name__ == '__main__':
    image_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated/img_000006_1.tif"
    mask_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks/img_000006_1_mask.tif"
    weight_map_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights/img_000006_1_weights.npy"
    imshow_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 255}
    letter_list = list(chr(i) for i in range(97, 97 + 26))
    conclusion_list = []
    image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    weight_map_np = np.load(weight_map_path)
    image_tf = tf.expand_dims(tf.convert_to_tensor(image_np, tf.float32), axis=-1)
    mask_tf = tf.expand_dims(tf.convert_to_tensor(mask_np, tf.float32), axis=-1)
    weight_map_tf = tf.convert_to_tensor(weight_map_np, tf.float32)

    run_the_segment_show_original_photo = False
    run_the_segment_illumination_voodoo_tf = False
    run_the_segment_histogram_voodoo_tf = False
    run_the_segment_elasticdeform_tf = False
    run_the_segment_random_flip = False
    run_the_random_rotate = False
    run_the_random_rot90 = False
    run_the_random_zoom_shift = False

    run_the_segment_illumination_voodoo_param_cmp = False
    run_the_segment_histogram_voodoo_param_cmp = False
    run_the_segment_elasticdeform_param_cmp = False
    run_the_segment_gaussian_noise_param_cmp = False
    run_the_segment_gaussian_blur_param_cmp = False

    saved_figure_dir = "E:/ED_MS/Semester_3/Codes/MyProject/saved_figures"

    # show original photo
    if run_the_segment_show_original_photo:
        plt.imshow(image_np, **imshow_kwargs)
        plt.title("original")
        plt.tight_layout(pad=2.5)
        plt.show()

    # illumination voodoo - tf version
    if run_the_segment_illumination_voodoo_tf:
        np.random.seed(0)
        with Timer("illumination_voodoo numpy ver"):
            image_np_illumination_voodoo = illumination_voodoo(image_np, num_control_points=3)
        plt.imshow(image_np_illumination_voodoo, **imshow_kwargs)
        plt.title("illumination_voodoo, num_control_points = 3")
        plt.show()

        np.random.seed(0)
        with Timer("illumination_voodoo tf ver"):
            image_tf_illumination_voodoo = illumination_voodoo_tf(image_tf, num_control_points=3)

        plt.imshow(image_tf_illumination_voodoo.numpy().squeeze(), **imshow_kwargs)
        plt.title("illumination_voodoo_tf, num_control_points = 3")
        plt.show()

    # illumination voodoo - parameter `num_control_points`
    if run_the_segment_illumination_voodoo_param_cmp:
        num_control_points_list = [4, 5, 6]
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0][0].imshow(image_np, **imshow_kwargs)
        axes[0][0].set_title("(a) original image", y=0, pad=-20, verticalalignment="top")
        for index, num_control_points in enumerate(num_control_points_list):
            index += 1
            row = index // 2
            column = index % 2
            np.random.seed(0)
            image_tf_illumination_voodoo = illumination_voodoo_tf(image_tf, num_control_points=num_control_points)
            axes[row][column].imshow(image_tf_illumination_voodoo.numpy().squeeze(), **imshow_kwargs)
            axes[row][column].set_title("({}) num_control_points = {}".format(letter_list[index], num_control_points),
                                        y=0, pad=-20, verticalalignment="top")
        f.suptitle("Illumination Voodoo", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "illumination_voodoo_param_cmp.jpg"))
        # plt.show()
        conclusion_list.append("For illumination_voodoo, num_control_points = 4 is a considerable configuration")

    # histogram voodoo - tf version
    if run_the_segment_histogram_voodoo_tf:
        np.random.seed(0)
        with Timer("histogram_voodoo np ver"):
            image_np_histogram_voodoo = histogram_voodoo(image_np, 3)
        plt.imshow(image_np_histogram_voodoo, **imshow_kwargs)
        plt.title("histogram_voodoo, num_control_points = 3")
        plt.show()

        np.random.seed(0)
        with Timer("histogram_voodoo tf ver"):
            image_tf_histogram_voodoo = histogram_voodoo_tf(image_tf, 3)
        plt.imshow(image_tf_histogram_voodoo.numpy().squeeze(), **imshow_kwargs)
        plt.title("histogram_voodoo_tf, num_control_points = 3")
        plt.show()

    # histogram voodoo - parameter `num_control_points`
    if run_the_segment_histogram_voodoo_param_cmp:
        num_control_points_list = [2, 6, 10]
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0][0].imshow(image_np, **imshow_kwargs)
        axes[0][0].set_title("(a) original", y=0, pad=-20, verticalalignment="top")
        for index, num_control_points in enumerate(num_control_points_list):
            index += 1
            row = index // 2
            column = index % 2
            image_np_histogram_voodoo = histogram_voodoo(image_np, num_control_points)
            axes[row][column].imshow(image_np_histogram_voodoo, **imshow_kwargs)
            axes[row][column].set_title("({}) num_control_points = {}".format(letter_list[index],
                                                                              num_control_points),
                                        y=0, pad=-20, verticalalignment="top")
        f.suptitle("Histogram Voodoo", fontsize=16)
        # plt.show()
        f.savefig(os.path.join(saved_figure_dir, "histogram_voodoo_param_cmp.jpg"))
        conclusion_list.append("For parameter num_control_points does not make much different to the output")

    # elasticdeform - tf version
    if run_the_segment_elasticdeform_tf:
        image_np_elasticdeform, mask_np_elasticdeform = elasticdeform_np([image_np, mask_np])
        plt.imshow(image_np_elasticdeform, **imshow_kwargs)
        plt.title("elasticdeform: image")
        plt.show()
        plt.imshow(mask_np_elasticdeform, **imshow_kwargs)
        plt.title("elasticdeform: mask")
        plt.show()

        image_tf_elasticdeform = elasticdeform_tf(image_tf)
        mask_tf_elasticdeform = elasticdeform_tf(mask_tf)
        plt.imshow(image_tf_elasticdeform.numpy().squeeze(), **imshow_kwargs)
        plt.title("elasticdeform_tf: image")
        plt.show()
        plt.imshow(mask_tf_elasticdeform.numpy().squeeze(), **imshow_kwargs)
        plt.title("elasticdeform_tf: mask")

    # elasticdeform - parameter `sigma`, `points`, and `mode`
    if run_the_segment_elasticdeform_param_cmp:
        sigma_list = [15, 20, 25]
        points_list = [3, 6, 9]
        # sigma_list = [25]*3
        # points_list = [3]*3

        for sigma_index, sigma in enumerate(sigma_list):
            f, axes = plt.subplots(2, 2, figsize=(20, 10))
            outer = gridspec.GridSpec(2, 2, figure=f, wspace=0.2, hspace=0.2)
            inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.1, hspace=0.1)

            ax = plt.Subplot(f, inner[0])
            ax.imshow(image_tf.numpy().squeeze(), **imshow_kwargs)
            f.add_subplot(ax)

            ax = plt.Subplot(f, inner[1])
            ax.imshow(mask_tf.numpy().squeeze(), **imshow_kwargs)
            f.add_subplot(ax)

            subplot_title_list = ["(a) original"]

            for zoom in [1.2, None]:
                for points_index, points in enumerate(points_list):
                    points_index += 1

                    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[points_index], wspace=0.1,
                                                             hspace=0.1)
                    # np.random.seed(0)
                    image_tf_elasticdeform, mask_tf_elasticdeform = elasticdeform_tf(image_tf, mask_tf, sigma, points,
                                                                                     mode="constant", zoom=zoom)

                    ax = plt.Subplot(f, inner[0])
                    ax.imshow(image_tf_elasticdeform.numpy().squeeze(), **imshow_kwargs)
                    f.add_subplot(ax)

                    ax = plt.Subplot(f, inner[1])
                    ax.imshow(mask_tf_elasticdeform.numpy().squeeze(), **imshow_kwargs)
                    f.add_subplot(ax)

                    subplot_title_list.append("({}) sigma = {}, points = {}".format(letter_list[points_index],
                                                                                    sigma, points))

                for ii, ax in enumerate(f.get_axes()):
                    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
                    if ii < 4:
                        ax.set_title(subplot_title_list[ii], y=0, pad=-20, verticalalignment="top")
                fig_suptitle = "Elasticdeform (zoom = {})".format(zoom) if zoom else "Elasticdeform"
                f.suptitle(fig_suptitle, fontsize=16)
                if zoom:
                    file_name = "Elasticdeform-zoom_{}-sigma_{}.jpg".format(zoom, sigma)
                else:
                    file_name = "Elasticdeform-no_zoom-sigma_{}.jpg".format(sigma)
                f.savefig(os.path.join(saved_figure_dir, file_name))
                f.show()

    if run_the_segment_gaussian_noise_param_cmp:
        sigma_list = [0.05, 0.1, 0.15]
        f, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0][0].imshow(image_tf.numpy().squeeze(), **imshow_kwargs)
        axes[0][0].set_title("original", y=0, pad=-20, verticalalignment="top")
        for index, sigma in enumerate(sigma_list):
            index += 1
            row = index // 2
            column = index % 2
            image_tf_gaussian_noise = gaussian_noise_tf(image_tf / 255.0, sigma=sigma)
            axes[row][column].imshow(image_tf_gaussian_noise.numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[row][column].set_title("({}) sigma = {}".format(letter_list[index], sigma), y=0, pad=-20,
                                        verticalalignment="top")
        f.suptitle("Gaussian Noise", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "Gaussian_Noise.jpg"))
        f.show()

    if run_the_segment_gaussian_blur_param_cmp:
        sigma_list = [1, 2, 3]
        f, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)

        axes[0][0].imshow(image_tf.numpy().squeeze(), **imshow_kwargs)
        axes[0][0].set_title("original", y=0, pad=-20, verticalalignment="top")
        for index, sigma in enumerate(sigma_list):
            index += 1
            row = index // 2
            column = index % 2
            image_tf_gaussian_blur = gaussian_blur_tf(image_tf / 255.0, sigma=sigma, filter_shape=(3, 3))
            axes[row][column].imshow(image_tf_gaussian_blur.numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[row][column].set_title("({}) sigma = {}".format(letter_list[index], sigma), y=0, pad=-20,
                                        verticalalignment="top")
        f.suptitle("Gaussian Blur", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "Gaussian_Blur.jpg"))
        f.show()

    if run_the_segment_random_flip:
        image_tf_flipped, mask_tf_flipped = random_flip_tf(image_tf, mask_tf)

        f, axes = plt.subplots(1, 2)
        axes[0].imshow(image_tf_flipped.numpy().squeeze(), **imshow_kwargs)
        axes[1].imshow(mask_tf_flipped.numpy().squeeze(), **imshow_kwargs)
        f.suptitle("Random Flip", fontsize=16)
        f.show()

    if run_the_random_rotate:
        image_tf_random_rotate, mask_tf_random_rotate, weight_map_tf_random_rotate = random_rotate_tf(image_tf,
                                                                                                      mask_tf,
                                                                                                      weight_map_tf,
                                                                                                      max_angle=np.pi / 12)
        print(tf.reduce_max(image_tf), tf.reduce_max(image_tf_random_rotate))
        f, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        axes[0].imshow(image_tf_random_rotate.numpy().squeeze(), **imshow_kwargs)
        axes[1].imshow(mask_tf_random_rotate.numpy().squeeze(), **imshow_kwargs)
        axes[2].imshow(weight_map_tf_random_rotate.numpy().squeeze())
        f.suptitle("Random Rotate", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "Random_Rotate.jpg"))
        # f.show()

    if run_the_random_rot90:
        image_tf_random_rot90, mask_tf_random_rot90, weight_map_tf_random_rot90 = random_rot90_tf(image_tf, mask_tf,
                                                                                                  weight_map_tf)
        f, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        axes[0].imshow(image_tf_random_rot90.numpy().squeeze(), **imshow_kwargs)
        axes[1].imshow(mask_tf_random_rot90.numpy().squeeze(), **imshow_kwargs)
        axes[2].imshow(weight_map_tf_random_rot90.numpy().squeeze())
        f.suptitle("Random Rot90", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "Random_Rot90.jpg"))
        f.show()

    if run_the_random_zoom_shift:
        image_tf_random_zoom_shift, mask_tf_random_zoom_shift, weight_map_tf_random_zoom_shift = \
            random_zoom_and_shift_tf(image_tf, mask_tf, weight_map_tf, beta=0.5)

        f, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        axes[0].imshow(image_tf_random_zoom_shift.numpy().squeeze(), **imshow_kwargs)
        axes[1].imshow(mask_tf_random_zoom_shift.numpy().squeeze(), **imshow_kwargs)
        axes[2].imshow(weight_map_tf_random_zoom_shift.numpy().squeeze())
        f.suptitle("Random Zoom and Shift", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "Random_Zoom_Shift.jpg"))
        f.show()

    beta = 0.05
    lamb = 1 / beta
    x = np.linspace(0, 1, 500)
    y = 1 - np.exp(-lamb * x)
    plt.plot(x, y, label="β={}".format(beta))
    plt.show()
    print(y[len(y) // 2])
