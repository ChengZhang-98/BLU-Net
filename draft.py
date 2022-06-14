import glob
import os
import itertools

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          Dropout,
                          UpSampling2D,
                          Concatenate)
from keras.models import Model
from keras.utils import Sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_contracting_block(input_layer, filters, conv2d_params, dropout=0, name="Contracting"):
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
    conv1 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_1")(pool)
    conv2 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")(conv1)

    if dropout == 0:
        return conv2
    else:
        return Dropout(rate=dropout, name=name + "_Dropout")(conv2)


def get_expanding_block(input_layer, skip_layer, filters, conv2d_params, dropout=0, name="Expanding"):
    up = UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = Conv2D(filters, 2, **conv2d_params, name=name + "_Conv2D_1")(up)
    merge = Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])
    conv2 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")(merge)
    conv3 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_3")(conv2)

    if dropout == 0:
        return conv3
    else:
        return Dropout(rate=dropout, name=name + "_Dropout")(conv3)


def get_uncompiled_unet(input_size, final_activation, output_classes, dropout=0, levels=5):
    conv2d_parameters = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }
    inputs = Input(input_size, name="true_input")
    filters = 64

    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_1")(inputs)
    conv = Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_2")(conv)

    level = 0
    contracting_outputs = [conv]
    for level in range(1, levels):
        filters *= 2
        contracting_outputs.append(
            get_contracting_block(input_layer=contracting_outputs[-1],
                                  filters=filters,
                                  conv2d_params=conv2d_parameters,
                                  dropout=dropout,
                                  name="Level{}_Contracting".format(level)
                                  )
        )

    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = get_expanding_block(input_layer=expanding_output,
                                               skip_layer=contracting_outputs.pop(),
                                               filters=filters,
                                               conv2d_params=conv2d_parameters,
                                               dropout=dropout,
                                               name="Level{}_Expanding".format(level))

    output = Conv2D(output_classes, 1, activation=final_activation, name="true_output")(expanding_output)

    unet_model = Model(inputs=inputs, outputs=output, name="Uncompiled_Unet")

    return unet_model


def get_paired_data_df(image_dir, image_type, mask_dir, mask_type, dataset):
    image_df = pd.DataFrame({"image": glob.glob(os.path.join(image_dir, "*." + image_type.lower()))})
    image_df.loc[:, "id"] = image_df.loc[:, "image"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    if mask_dir is None:
        image_df.loc[:, "mask"] = None
        paired_df = image_df.copy()
    else:

        mask_df = pd.DataFrame({"mask": glob.glob(os.path.join(mask_dir, "*." + mask_type.lower()))})
        if dataset.lower() == "training_2d":
            mask_df.loc[:, "id"] = mask_df.loc[:, "mask"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        else:
            mask_df.loc[:, "id"] = mask_df.loc[:, "mask"].apply(lambda x: os.path.splitext(os.path.basename(x))[0][:-5])

        paired_df = pd.merge(image_df, mask_df, how="inner", on=["id"])

    paired_df = paired_df.sort_values(by=["image"], ignore_index=True)
    return paired_df


class DataGenerator(Sequence):
    def __init__(self, batch_size, dataset,
                 image_dir,
                 image_type,
                 mask_dir,
                 mask_type,
                 target_size,
                 transforms,
                 seed):
        super(DataGenerator, self).__init__()
        self.batch_size = batch_size
        self.target_size = target_size
        self.transforms = transforms
        self.seed = seed
        if mask_dir is not None:
            self.train = True
        else:
            self.train = False

        self.data_df = get_paired_data_df(image_dir=image_dir, image_type=image_type,
                                          mask_dir=mask_dir, mask_type=mask_type,
                                          dataset=dataset)
        if seed is None:
            self.seed_gen = itertools.count(start=0, step=1)
        else:
            self.seed_gen = itertools.count(start=seed, step=1)
            self.data_df = self.data_df.sample(frac=1, random_state=seed, ignore_index=True)

        self.batch_counts = int(len(self.data_df) / batch_size)

    def __getitem__(self, index):
        batch_df = self.data_df.iloc[self.batch_size * index:self.batch_size * (index + 1), :]
        image_batch = []
        mask_batch = []
        for i, row in enumerate(batch_df.itertuples()):
            image_i = load_an_image(row.image)

            # load an image
            if self.train:
                mask_i = load_an_image(row.mask)
            else:
                mask_i = None

            # data preprocessing
            image_i, mask_i = padding_crop_and_rescale(image_i, mask_i, target_size=self.target_size, foreground=0)

            # *: data augmentation here

            # append to arrays
            image_batch.append(tf.expand_dims(image_i, axis=0))
            if self.train:
                mask_batch.append(tf.expand_dims(mask_i, axis=0))
            else:
                mask_batch.append(None)

        image_batch = tf.concat(image_batch, axis=0)

        if self.train:
            mask_batch = tf.concat(mask_batch, axis=0)

        return image_batch, mask_batch

    def __len__(self):
        return self.batch_counts

    def on_epoch_end(self):
        self.data_df = self.data_df.sample(frac=1, random_state=next(self.seed_gen))


def load_an_image(image_path):
    """
    load an image to a tensor
    :param image_path: path to grayscale image
    :return: HxWx1 tensor
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = tf.convert_to_tensor(image, dtype=tf.int8)
    image = tf.expand_dims(image, axis=-1)
    return image


def padding_crop_and_rescale(image, mask, target_size, foreground=0):
    process_mask = mask is not None

    if process_mask:
        assert image.shape == mask.shape, "image.shape != mask.shape"

    # cropped_height = max(image.shape[-3], target_size[0])
    # cropped_width = max(image.shape[-2], target_size[1])

    image = tf.image.resize_with_crop_or_pad(image, target_height=target_size[0], target_width=target_size[1])
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0

    if process_mask:
        mask = tf.image.resize_with_crop_or_pad(mask, target_height=target_size[0], target_width=target_size[1])

        if foreground == 0:
            mask = tf.cast(mask < tf.math.reduce_mean(mask), tf.float32)
        elif foreground == 1:
            mask = tf.cast(mask > tf.math.reduce_mean(mask), tf.float32)
        else:
            raise RuntimeError("resize_and_rescale, unsupported foreground: {}".format(foreground))

    return image, mask


def data_augmentation(image, mask):
    pass


def main():
    seed = 1
    batch_size = 4
    target_size = (128, 128)

    unet = get_uncompiled_unet((target_size[0], target_size[1], 1),
                               final_activation="sigmoid",
                               output_classes=1,
                               levels=5)
    unet.load_weights(filepath="./checkpoints/trained_weights/unet_moma_seg_multisets.hdf5")

    # DIC
    # image_dir = "../../Dataset/DIC_Set/DIC_Set1_Annotated"
    # image_type = "tif"
    # mask_dir = "../../Dataset/DIC_Set/DIC_Set1_Masks"
    # mask_type = "tif"

    # Training_2D
    image_dir = "../../Dataset/training_2D/training/segmentation_set/img"
    image_type = "png"
    mask_dir = "../../Dataset/training_2D/training/segmentation_set/seg"
    mask_type = "png"

    data_gen = DataGenerator(batch_size=batch_size, dataset="training_2d",
                             image_dir=image_dir, image_type=image_type,
                             mask_dir=mask_dir, mask_type=mask_type,
                             target_size=target_size, transforms=None, seed=None)

    fig_size = (16, 8)
    fig_image, axes_image = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)
    fig_predicted, axes_predicted = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)
    fig_mask, axes_mask = plt.subplots(2, batch_size, figsize=fig_size, dpi=150)

    for row, (image_batch, mask_batch) in enumerate(data_gen):
        if row >= 2:
            break
        predicted_mask_batch = unet.predict_step(image_batch)
        for col in range(batch_size):
            image_col = image_batch[col, ...].numpy().squeeze()
            axes_image[row, col].imshow(image_col, cmap='gray', vmin=0, vmax=1)

            mask_col = mask_batch[col, ...].numpy().squeeze()
            axes_mask[row, col].imshow(mask_col, cmap='gray', vmin=0, vmax=1)

            predicted_mask_col = predicted_mask_batch[col, ...].numpy().squeeze()
            axes_predicted[row, col].imshow(predicted_mask_col, cmap="gray", vmin=0, vmax=1)
            axes_predicted[row, col].set_title("predicted_mask")

    plt.show()


if __name__ == '__main__':
    main()
