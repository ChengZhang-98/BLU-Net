import os

import keras.backend
import numpy as np
import tensorflow as tf
from keras import metrics, initializers
from keras import layers
from keras import regularizers
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy, BinaryIoU, OneHotIoU, IoU
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from residual_binarization import BinarySignActivation, BinaryConv2D, BinarySeparableConv2D
from training_utils import CustomBinaryIoU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _get_contracting_block(input_layer, conv2d_layer, filters, conv2d_params, dropout=0, name="Contracting",
                           regularizer_factor=0):
    if conv2d_layer is layers.Conv2D:
        conv_name = "_Conv2D"
    elif conv2d_layer is layers.SeparableConv2D:
        conv_name = "_SeparableConv2D"
    else:
        conv_name = "UnknownConv2D"

    pool = layers.MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
    conv1 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_1",
                         kernel_regularizer=regularizers.L2(regularizer_factor))(pool)
    conv2 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_2",
                         kernel_regularizer=regularizers.L2(regularizer_factor))(conv1)

    if dropout == 0:
        return conv2
    else:
        return layers.Dropout(rate=dropout, name=name + "_Dropout")(conv2)


def _get_expanding_block(input_layer, skip_layer, conv2d_layer, filters, conv2d_params, dropout=0, name="Expanding",
                         regularizer_factor=0):
    if conv2d_layer is layers.Conv2D:
        conv_name = "_Conv2D"
    elif conv2d_layer is layers.SeparableConv2D:
        conv_name = "_SeparableConv2D"
    else:
        conv_name = "UnknownConv2D"
    up = layers.UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
    conv1 = conv2d_layer(filters, 2, **conv2d_params, name=name + conv_name + "_1",
                         kernel_regularizer=regularizers.L2(regularizer_factor))(up)
    merge = layers.Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])
    conv2 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_2",
                         kernel_regularizer=regularizers.L2(regularizer_factor))(merge)
    conv3 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_3",
                         kernel_regularizer=regularizers.L2(regularizer_factor))(conv2)

    if dropout == 0:
        return conv3
    else:
        return layers.Dropout(rate=dropout, name=name + "_Dropout")(conv3)


def get_uncompiled_unet(input_size, final_activation, output_classes, dropout=0, num_levels=5, regularizer_factor=0):
    conv2d_parameters = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }
    inputs = layers.Input(input_size, name="input")
    filters = 64

    conv = layers.Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_1")(inputs)
    conv = layers.Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_2")(conv)

    level = 0
    contracting_outputs = [conv]
    for level in range(1, num_levels):
        filters *= 2
        contracting_outputs.append(
            _get_contracting_block(input_layer=contracting_outputs[-1],
                                   filters=filters,
                                   conv2d_layer=layers.Conv2D,
                                   conv2d_params=conv2d_parameters,
                                   dropout=dropout,
                                   name="Level{}_Contracting".format(level),
                                   regularizer_factor=regularizer_factor
                                   )
        )

    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = _get_expanding_block(input_layer=expanding_output,
                                                skip_layer=contracting_outputs.pop(),
                                                filters=filters,
                                                conv2d_layer=layers.Conv2D,
                                                conv2d_params=conv2d_parameters,
                                                dropout=dropout,
                                                name="Level{}_Expanding".format(level),
                                                regularizer_factor=regularizer_factor)

    output = layers.Conv2D(output_classes, 1, activation=final_activation, name="output")(expanding_output)

    unet_model = Model(inputs=inputs, outputs=output, name="Uncompiled_Unet")

    return unet_model


def get_compiled_unet(input_size, num_levels=5, final_activation="sigmoid", pretrained_weights=None, learning_rate=1e-4,
                      regularizer_factor=0):
    unet_model = get_uncompiled_unet(input_size, final_activation=final_activation, output_classes=1, dropout=0,
                                     num_levels=num_levels, regularizer_factor=regularizer_factor)
    bce_loss_from_logits = final_activation != "sigmoid"
    unet_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss=BinaryCrossentropy(name="weighted_binary_crossentropy", from_logits=bce_loss_from_logits),
                       metrics=[CustomBinaryIoU(threshold=0.5, name="binary_IoU"),
                                BinaryAccuracy(threshold=0.5, name="binary_accuracy")])
    # keras.metrics.BinaryIoU can not deal with interpolated values between 0.0 and 1.0
    # metrics=[BinaryIoU(name="binary_IoU", target_class_ids=[1], threshold=0.5)])
    if pretrained_weights is not None:
        unet_model.load_weights(filepath=pretrained_weights)

    return unet_model


def get_uncompiled_lightweight_unet(input_size, final_activation, output_classes, dropout=0, num_levels=5):
    conv2d_parameters = {
        "activation": "relu",
        "padding": "same",
        "kernel_initializer": "he_normal",
    }
    inputs = layers.Input(input_size, name="input")
    filters = 64

    conv = layers.Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_1")(inputs)
    conv = layers.Conv2D(filters, 3, **conv2d_parameters, name="Level0_Conv2D_2")(conv)

    level = 0
    contracting_outputs = [conv]
    for level in range(1, num_levels):
        filters *= 2
        contracting_outputs.append(
            _get_contracting_block(input_layer=contracting_outputs[-1],
                                   filters=filters,
                                   conv2d_layer=layers.SeparableConv2D,
                                   conv2d_params=conv2d_parameters,
                                   dropout=dropout,
                                   name="Level{}_Lightweight_Contracting".format(level)
                                   )
        )

    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = _get_expanding_block(input_layer=expanding_output,
                                                skip_layer=contracting_outputs.pop(),
                                                filters=filters,
                                                conv2d_layer=layers.SeparableConv2D,
                                                conv2d_params=conv2d_parameters,
                                                dropout=dropout,
                                                name="Level{}_Lightweight_Expanding".format(level))

    output = layers.Conv2D(output_classes, 1, activation=final_activation, name="output")(expanding_output)

    lw_unet_model = Model(inputs=inputs, outputs=output, name="Uncompiled_Unet")

    return lw_unet_model


def get_compiled_lightweight_unet(input_size, num_levels=5, output_classes=1, learning_rate=1e-3,
                                  pretrained_weight=None):
    lw_unet = get_uncompiled_lightweight_unet(input_size=input_size,
                                              final_activation="sigmoid",
                                              output_classes=output_classes, num_levels=num_levels)
    lw_unet.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss=BinaryCrossentropy(name="weighted_binary_crossentropy"),
                    metrics=[CustomBinaryIoU(threshold=0.5, name="binary_IoU"),
                             BinaryAccuracy(threshold=0.5, name="binary_accuracy")])
    if pretrained_weight is not None:
        lw_unet.load_weights(filepath=pretrained_weight)
    return lw_unet


def _get_binary_contracting_block(inputs, conv2d_layer: [BinaryConv2D, BinarySeparableConv2D],
                                  num_activation_residual_levels, conv_residual_level_dict, filters, padding,
                                  kernel_initializer_seed, name="BinaryContractingBlock"):
    if conv2d_layer is BinaryConv2D:
        conv_name = "_BinaryConv2D"
    elif conv2d_layer is BinarySeparableConv2D:
        conv_name = "_BinarySeparableConv2D"
    else:
        conv_name = "UnknownConv2D"

    pool = layers.MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(inputs)
    conv1 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_1")(pool)
    conv1 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_1")(conv1)
    conv2 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_2")(conv1)
    conv2 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_2")(conv2)
    return conv2


def _get_binary_expanding_block(inputs, skip_connection, conv2d_layer: [BinaryConv2D, BinarySeparableConv2D],
                                num_activation_residual_levels, conv_residual_level_dict, filters, padding,
                                kernel_initializer_seed, name="BinaryExpandingBlock"):
    if conv2d_layer is BinaryConv2D:
        conv_name = "_BinaryConv2D"
    elif conv2d_layer is BinarySeparableConv2D:
        conv_name = "_BinarySeparableConv2D"
    else:
        conv_name = "UnknownConv2D"
    up = layers.UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(inputs)
    conv1 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=2, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_1")(up)
    conv1 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_1")(conv1)
    merge = layers.Concatenate(axis=3, name=name + "_Concatenate")([skip_connection, conv1])
    conv2 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_2")(merge)
    conv2 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_2")(conv2)
    conv3 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_3")(conv2)
    conv3 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_3")(conv3)
    return conv3


def get_uncompiled_binary_unet(input_size, num_activation_residual_levels, num_conv_residual_levels,
                               output_classes, num_levels=5, conv_kernel_initializer_seed=1):
    inputs = layers.Input(input_size, name="input")
    filters = 64

    conv = BinaryConv2D(num_residual_levels=num_conv_residual_levels, filters=filters, kernel_size=3,
                        padding="SAME", name="Level0_BinaryConv2D_1")(inputs)
    conv = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                name="Level0_BinarySignActivation_1")(conv)
    conv = BinaryConv2D(num_residual_levels=num_conv_residual_levels, filters=filters, kernel_size=3,
                        padding="SAME", name="Level0_BinaryConv2D_2")(conv)
    conv = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                name="Level0_BinarySignActivation_2")(conv)

    level = 0
    contracting_outputs = [conv]
    for level in range(1, num_levels):
        filters *= 2
        contracting_outputs.append(
            _get_binary_contracting_block(inputs=contracting_outputs[-1],
                                          conv2d_layer=BinaryConv2D,
                                          num_activation_residual_levels=num_activation_residual_levels,
                                          conv_residual_level_dict=dict(num_residual_levels=num_conv_residual_levels),
                                          filters=filters,
                                          padding="SAME",
                                          kernel_initializer_seed=conv_kernel_initializer_seed,
                                          name="Level{}_BinaryContracting".format(level))
        )

    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = int(filters / 2)
        expanding_output = _get_binary_expanding_block(inputs=expanding_output,
                                                       skip_connection=contracting_outputs.pop(),
                                                       conv2d_layer=BinaryConv2D,
                                                       num_activation_residual_levels=num_activation_residual_levels,
                                                       conv_residual_level_dict=dict(
                                                           num_residual_levels=num_conv_residual_levels),
                                                       filters=filters,
                                                       padding="SAME",
                                                       kernel_initializer_seed=conv_kernel_initializer_seed,
                                                       name="Level{}_BinaryExpanding".format(level))
    output = BinaryConv2D(num_residual_levels=num_conv_residual_levels, filters=output_classes, kernel_size=1,
                          name="logit_output")(expanding_output)
    # output = BinarySignActivation(num_residual_levels=num_activation_residual_levels)(output)
    output = layers.Activation(tf.nn.sigmoid, name="sigmoid_output")(output)

    binary_unet_model = Model(inputs=inputs, outputs=output, name="UncompiledBinaryUnet")
    return binary_unet_model


def get_compiled_binary_unet(input_size, num_activation_residual_levels=3, num_conv_residual_levels=3,
                             output_classes=1, num_levels=5, initializer_seed=1, learning_rate=1e-3,
                             pretrained_weight=None):
    binary_unet = get_uncompiled_binary_unet(input_size=input_size,
                                             num_activation_residual_levels=num_activation_residual_levels,
                                             num_conv_residual_levels=num_conv_residual_levels,
                                             output_classes=output_classes,
                                             num_levels=num_levels, conv_kernel_initializer_seed=initializer_seed)
    binary_unet.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss=BinaryCrossentropy(name="binary_crossentropy"),
                        metrics=[CustomBinaryIoU(threshold=0.5, name="binary_IoU"),
                                 BinaryAccuracy(threshold=0.5, name="binary_accuracy")])
    if pretrained_weight is not None:
        binary_unet.load_weights(filepath=pretrained_weight)
    return binary_unet


def get_uncompiled_binary_lightweight_unet(input_size,
                                           num_activation_residual_levels,
                                           num_conv_residual_levels,
                                           num_residual_levels_depthwise_filter,
                                           num_residual_levels_pointwise_filter,
                                           output_classes=1,
                                           num_levels=5,
                                           conv_kernel_initializer_seed=1):
    inputs = layers.Input(input_size, name="input")
    filters = 64

    conv = BinaryConv2D(num_residual_levels=num_conv_residual_levels, filters=filters, kernel_size=3,
                        padding="SAME", name="Level0_BinaryConv2D_1")(inputs)
    conv = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                name="Level0_BinarySignActivation_1")(conv)
    conv = BinaryConv2D(num_residual_levels=num_conv_residual_levels, filters=filters, kernel_size=3,
                        padding="SAME", name="Level0_BinaryConv2D_2")(conv)
    conv = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                name="Level0_BinarySignActivation_2")(conv)

    level = 0
    contracting_outputs = [conv]
    for level in range(1, num_levels):
        filters *= 2
        contracting_outputs.append(
            _get_binary_contracting_block(
                inputs=contracting_outputs[-1],
                conv2d_layer=BinarySeparableConv2D,
                num_activation_residual_levels=num_activation_residual_levels,
                conv_residual_level_dict=dict(
                    num_residual_levels_depthwise_filter=num_residual_levels_depthwise_filter,
                    num_residual_levels_pointwise_filter=num_residual_levels_pointwise_filter),
                filters=filters,
                padding="SAME",
                kernel_initializer_seed=conv_kernel_initializer_seed,
                name="Level{}_BinaryLightweight_Contracting".format(level)
            )
        )

    expanding_output = contracting_outputs.pop()
    while level > 0:
        level -= 1
        filters = filters // 2
        expanding_output = _get_binary_expanding_block(
            inputs=expanding_output,
            skip_connection=contracting_outputs.pop(),
            conv2d_layer=BinarySeparableConv2D,
            num_activation_residual_levels=num_activation_residual_levels,
            conv_residual_level_dict=dict(
                num_residual_levels_depthwise_filter=num_residual_levels_depthwise_filter,
                num_residual_levels_pointwise_filter=num_residual_levels_pointwise_filter
            ),
            filters=filters,
            padding="SAME",
            kernel_initializer_seed=conv_kernel_initializer_seed,
            name="Level{}_BinaryLightweight_Expanding".format(level)
        )

    output = BinaryConv2D(num_residual_levels=num_conv_residual_levels, filters=output_classes, kernel_size=1,
                          name="logit_output")(expanding_output)
    output = layers.Activation(tf.nn.sigmoid, name="sigmoid_output")(output)

    binary_unet_model = Model(inputs=inputs, outputs=output, name="UncompiledBinaryLightweightUnet")
    return binary_unet_model


def get_compiled_binary_lightweight_unet(input_size,
                                         num_activation_residual_levels=3,
                                         num_conv_residual_levels=3,
                                         num_depthwise_conv_residual_levels=3,
                                         num_pointwise_conv_residual_levels=3,
                                         output_classes=1,
                                         num_levels=5,
                                         initializer_seed=1,
                                         learning_rate=1e-3,
                                         pretrained_weight=None):
    binary_lightweight_unet = get_uncompiled_binary_lightweight_unet(
        input_size=input_size,
        num_activation_residual_levels=num_activation_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        num_residual_levels_depthwise_filter=num_depthwise_conv_residual_levels,
        num_residual_levels_pointwise_filter=num_pointwise_conv_residual_levels,
        output_classes=output_classes,
        num_levels=num_levels,
        conv_kernel_initializer_seed=initializer_seed
    )

    binary_lightweight_unet.compile(optimizer=Adam(learning_rate=learning_rate),
                                    loss=BinaryCrossentropy(name="binary_crossentropy"),
                                    metrics=[CustomBinaryIoU(threshold=0.5, name="binary_IoU"),
                                             BinaryAccuracy(threshold=0.5, name="binary_accuracy")])

    if pretrained_weight is not None:
        binary_lightweight_unet.load_weights(pretrained_weight)

    return binary_lightweight_unet


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

    x_rand = tf.random.uniform((batch_size, *target_size, 1))
    lw_unet = get_compiled_lightweight_unet(input_size=(*target_size, 1), num_levels=5)
    # y_lw_unet = lw_unet(x_rand, training=False)
    # unet = get_compiled_unet(input_size=(*target_size, 1), num_levels=4)
    # y_unet = unet(x_rand, training=False)

    # unet = get_compiled_unet((*target_size, 1), num_levels=5)
    # lightweight_unet = get_compiled_lightweight_unet((*target_size, 1), 5)

    # binary_unet = get_compiled_binary_unet((*target_size, 1))

    binary_lightweight_unet = get_compiled_binary_lightweight_unet((*target_size, 1))
    # y_blw_unet = binary_lightweight_unet(x_rand, training=False)

    from residual_binarization import transfer_lightweight_unet_weights_to_binary_lightweight_unet

    binary_lightweight_unet = transfer_lightweight_unet_weights_to_binary_lightweight_unet(lw_unet,
                                                                                           binary_lightweight_unet)
    y_blw_unet = binary_lightweight_unet(x_rand, training=False)
