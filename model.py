import os

import keras.backend
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.layers import (Input,
                          Conv2D,
                          MaxPooling2D,
                          Dropout,
                          UpSampling2D,
                          Concatenate, BatchNormalization, LeakyReLU, Flatten, Dense)
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, BinaryIoU
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_addons as tfa
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _get_contracting_block(input_layer, filters, conv2d_params, dropout=0, name="Contracting"):
    pool = MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
    conv1 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_1")(pool)
    conv2 = Conv2D(filters, 3, **conv2d_params, name=name + "_Conv2D_2")(conv1)

    if dropout == 0:
        return conv2
    else:
        return Dropout(rate=dropout, name=name + "_Dropout")(conv2)


def _get_expanding_block(input_layer, skip_layer, filters, conv2d_params, dropout=0, name="Expanding"):
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
            _get_contracting_block(input_layer=contracting_outputs[-1],
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
        expanding_output = _get_expanding_block(input_layer=expanding_output,
                                                skip_layer=contracting_outputs.pop(),
                                                filters=filters,
                                                conv2d_params=conv2d_parameters,
                                                dropout=dropout,
                                                name="Level{}_Expanding".format(level))

    output = Conv2D(output_classes, 1, activation=final_activation, name="true_output")(expanding_output)

    unet_model = Model(inputs=inputs, outputs=output, name="Uncompiled_Unet")

    return unet_model


def get_compiled_unet(input_size, levels, final_activation="sigmoid", pretrained_weights=None, learning_rate=1e-4):
    unet_model = get_uncompiled_unet(input_size, final_activation=final_activation, output_classes=1, dropout=0,
                                     levels=levels)
    bce_loss_from_logits = final_activation != "sigmoid"
    unet_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss=BinaryCrossentropy(name="weighted_binary_crossentropy", from_logits=bce_loss_from_logits),
                       metrics=[BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                                BinaryIoU(name="binary_IoU", target_class_ids=[1], threshold=0.5)])
    if pretrained_weights is not None:
        unet_model.load_weights(filepath=pretrained_weights)

    return unet_model


def _get_conv_block(x, filters, activation,
                    kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True,
                    use_bn=False, use_dropout=False, drop_value=0.5):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = Dropout(drop_value)(x)
    return x


def get_discriminator(input_size, dropout=0.2):
    """
    Refer to https://keras.io/examples/generative/wgan_gp/
    :param input_size:
    :param dropout:
    :return:
    """
    inputs = Input(input_size)

    x = _get_conv_block(inputs, 64, kernel_size=(5, 5), strides=(2, 2),
                        use_bn=False, use_bias=True, activation=LeakyReLU(0.2),
                        use_dropout=False, drop_value=0.3)
    x = _get_conv_block(x, 128, kernel_size=(5, 5), strides=(2, 2),
                        use_bn=False, activation=LeakyReLU(0.2), use_bias=True,
                        use_dropout=True, drop_value=0.3)
    x = _get_conv_block(x, 256, kernel_size=(5, 5), strides=(2, 2),
                        use_bn=False, activation=LeakyReLU(0.2), use_bias=True,
                        use_dropout=True, drop_value=0.3)
    x = _get_conv_block(x, 512, kernel_size=(5, 5),
                        strides=(2, 2), use_bn=False, activation=LeakyReLU(0.2),
                        use_bias=True, use_dropout=True, drop_value=0.3)

    # *: one more conv_block than https://keras.io/examples/generative/wgan_gp/
    x = _get_conv_block(x, 1024, kernel_size=(5, 5),
                        strides=(2, 2), use_bn=False, activation=LeakyReLU(0.2),
                        use_bias=True, use_dropout=False, drop_value=0.3)

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    y = Dense(1)(x)

    return Model(inputs, y, name="discriminator")


class GAN:
    # https://www.tensorflow.org/tutorials/generative/pix2pix
    def __init__(self, discriminator, generator):
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="train-generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="train-discriminator_loss")
        self.metric_binary_accuracy = keras.metrics.BinaryAccuracy("val-binary_accuracy")
        self.metric_binary_IoU = keras.metrics.BinaryIoU(name="val-binary_IoU", target_class_ids=[1])

        self.d_optimizer, self.g_optimizer = None, None
        self.loss_fn_adversarial, self.loss_fn_segmentation = None, None
        self.lamda = 1

    def reset_metric_states(self):
        self.disc_loss_tracker.reset_state()
        self.gen_loss_tracker.reset_state()
        self.metric_binary_accuracy.reset_state()
        self.metric_binary_IoU.reset_state()

    def compile(self, d_optimizer, g_optimizer, loss_fn_adversarial, loss_fn_segmentation, lamda=1.0):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn_adversarial = loss_fn_adversarial
        self.loss_fn_segmentation = loss_fn_segmentation
        self.lamda = lamda

    def train_step(self, image_batch, real_mask_batch, train_g=True, train_d=True):
        if train_d:
            # train the discriminator
            generated_masks = self.generator(image_batch)
            couple_generated = tf.concat([image_batch, generated_masks], axis=-1)
            couple_real = tf.concat([image_batch, real_mask_batch], axis=-1)
            combined_images = tf.concat([couple_generated, couple_real], axis=0)
            # *: Note that 1 for generated couple, and 0 for real couple
            labels = tf.concat([tf.ones(image_batch.shape[0], 1), tf.zeros(real_mask_batch.shape[0], 1)],
                               axis=0)
            # ?: what's this. Tensorflow tutorial notes this is an important trick
            # https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#end-to-end_example_a_gan_training_loop_from_scratch
            labels += 0.05 * tf.random.uniform(labels.shape)

            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_images)
                d_loss = self.loss_fn_adversarial(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
            self.disc_loss_tracker.update_state(d_loss)
        else:
            d_loss = None
            generated_masks = None

        if train_g:
            # train the generator
            # here generator tries to fool the discriminator
            misleading_labels = tf.zeros((image_batch.shape[0], 1))
            with tf.GradientTape() as tape:
                predicted_masks = self.generator(image_batch)
                couple_generated = tf.concat([image_batch, predicted_masks], axis=-1)
                predictions = self.discriminator(couple_generated)
                # loss_seg = self.loss_fn_segmentation(image_batch, predicted_masks, weight_map_batch)
                loss_ad = self.loss_fn_adversarial(misleading_labels, predictions)
                # g_loss = loss_seg + self.lamda * loss_ad
                g_loss = loss_ad
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            self.gen_loss_tracker.update_state(g_loss)
        else:
            g_loss = None

        return d_loss, g_loss, generated_masks


if __name__ == '__main__':
    # *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"
    seed = None
    batch_size = 4
    target_size = (512, 512)
    weight_path = "./checkpoints/trained_weights/unet_agarpads_seg_evaluation2.hdf5"

    image_dir = "../../Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "../../Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "../../Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset = "DIC"

    discriminator = get_discriminator((512, 512, 1))
