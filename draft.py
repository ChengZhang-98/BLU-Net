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
from skimage import morphology as morph
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main1():
    from model import get_uncompiled_unet
    from data import DataGenerator
    seed = 1
    batch_size = 4
    target_size = (512, 512)

    unet = get_uncompiled_unet((target_size[0], target_size[1], 1),
                               final_activation="sigmoid",
                               output_classes=1,
                               levels=5)
    unet.load_weights(filepath="./checkpoints/trained_weights/unet_moma_seg_multisets.hdf5")

    # DIC
    image_dir = "../../Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "../../Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "../../Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset = "DIC"

    # Training_2D
    # image_dir = "../../Dataset/training_2D/training/segmentation_set/img"
    # image_type = "png"
    # mask_dir = "../../Dataset/training_2D/training/segmentation_set/seg"
    # mask_type = "png"

    data_gen = DataGenerator(batch_size=batch_size, dataset=dataset,
                             image_dir=image_dir, image_type=image_type,
                             mask_dir=mask_dir, mask_type=mask_type,
                             weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                             target_size=target_size, transforms=None, seed=None)

    fig_size = (16, 8)
    fig_image, axes_image = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)
    fig_image.suptitle("Cell Photo")
    fig_predicted, axes_predicted = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)
    fig_predicted.suptitle("Predicted Mask")
    fig_mask, axes_mask = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)
    fig_mask.suptitle("Ground Truth Mask")
    fig_weight, axes_weight = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)
    fig_weight.suptitle("Weights")

    for row, (image_batch, mask_weight_batch) in enumerate(data_gen):
        if row >= 2:
            break
        predicted_mask_batch = unet.predict_step(image_batch)
        for col in range(batch_size):
            image_col = image_batch[col, ...].numpy().squeeze()
            axes_image[row, col].imshow(image_col, cmap='gray', vmin=0, vmax=1)

            mask_weight_col = mask_weight_batch[col, ...]
            mask_col, weight_col = tf.unstack(mask_weight_col, axis=-1)
            mask_col = mask_col.numpy().squeeze()
            weight_col = weight_col.numpy().squeeze()
            axes_mask[row, col].imshow(mask_col, cmap='gray', vmin=0, vmax=1)
            axes_weight[row, col].imshow(weight_col, cmap="gray", vmin=0, vmax=100)

            predicted_mask_col = predicted_mask_batch[col, ...].numpy().squeeze()
            axes_predicted[row, col].imshow(predicted_mask_col, cmap="gray", vmin=0, vmax=1)
            axes_predicted[row, col].set_title("predicted_mask")

    plt.show()


if __name__ == '__main__':
    print("hi")
