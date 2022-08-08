import os

import keras.backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
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


def _get_contracting_block(input_layer, conv2d_layer, filters, conv2d_params,
                           dropout=0, name="Contracting",
                           regularizer_factor=0, channel_multiplier=1):
    if conv2d_layer is layers.Conv2D:
        conv_name = "_Conv2D"
        pool = layers.MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
        conv1 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_1",
                             kernel_regularizer=regularizers.L2(regularizer_factor))(pool)
        conv2 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_2",
                             kernel_regularizer=regularizers.L2(regularizer_factor))(conv1)
    elif conv2d_layer is layers.SeparableConv2D:
        conv_name = "_SeparableConv2D"
        pool = layers.MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(input_layer)
        conv1 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_1",
                             kernel_regularizer=regularizers.L2(regularizer_factor),
                             depth_multiplier=channel_multiplier)(pool)
        conv2 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_2",
                             kernel_regularizer=regularizers.L2(regularizer_factor),
                             depth_multiplier=channel_multiplier)(conv1)
    else:
        conv_name = "UnknownConv2D"
        raise RuntimeError("Unsupported Conv Layer")

    if dropout == 0:
        return conv2
    else:
        return layers.Dropout(rate=dropout, name=name + "_Dropout")(conv2)


def _get_expanding_block(input_layer, skip_layer, conv2d_layer, filters, conv2d_params, dropout=0, name="Expanding",
                         regularizer_factor=0, channel_multiplier=1):
    if conv2d_layer is layers.Conv2D:
        conv_name = "_Conv2D"
        up = layers.UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
        conv1 = conv2d_layer(filters, 2, **conv2d_params, name=name + conv_name + "_1",
                             kernel_regularizer=regularizers.L2(regularizer_factor))(up)
        merge = layers.Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])
        conv2 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_2",
                             kernel_regularizer=regularizers.L2(regularizer_factor))(merge)
        conv3 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_3",
                             kernel_regularizer=regularizers.L2(regularizer_factor))(conv2)
    elif conv2d_layer is layers.SeparableConv2D:
        conv_name = "_SeparableConv2D"
        up = layers.UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(input_layer)
        conv1 = conv2d_layer(filters, 2, **conv2d_params, name=name + conv_name + "_1",
                             kernel_regularizer=regularizers.L2(regularizer_factor),
                             depth_multiplier=channel_multiplier)(up)
        merge = layers.Concatenate(axis=3, name=name + "_Concatenate")([skip_layer, conv1])
        conv2 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_2",
                             kernel_regularizer=regularizers.L2(regularizer_factor))(merge)
        conv3 = conv2d_layer(filters, 3, **conv2d_params, name=name + conv_name + "_3",
                             kernel_regularizer=regularizers.L2(regularizer_factor),
                             depth_multiplier=channel_multiplier)(conv2)
    else:
        conv_name = "UnknownConv2D"
        raise RuntimeError("Unsupported Conv Layer")

    if dropout == 0:
        return conv3
    else:
        return layers.Dropout(rate=dropout, name=name + "_Dropout")(conv3)


def get_uncompiled_unet(input_size, final_activation="sigmoid", output_classes=1, dropout=0, num_levels=5,
                        regularizer_factor=0):
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


def get_uncompiled_lightweight_unet(input_size, final_activation="sigmoid", output_classes=1, dropout=0, num_levels=5,
                                    regularizer_factor=0, channel_multiplier=1):
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
                                   name="Level{}_Lightweight_Contracting".format(level),
                                   regularizer_factor=regularizer_factor,
                                   channel_multiplier=channel_multiplier
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
                                                name="Level{}_Lightweight_Expanding".format(level),
                                                regularizer_factor=regularizer_factor,
                                                channel_multiplier=channel_multiplier)

    output = layers.Conv2D(output_classes, 1, activation=final_activation, name="output")(expanding_output)

    lw_unet_model = Model(inputs=inputs, outputs=output, name="Uncompiled_Unet")

    return lw_unet_model


def get_compiled_lightweight_unet(input_size, num_levels=5, output_classes=1, learning_rate=1e-3,
                                  pretrained_weight=None, regularizer_factor=0, channel_multiplier=1):
    lw_unet = get_uncompiled_lightweight_unet(input_size=input_size,
                                              final_activation="sigmoid",
                                              output_classes=output_classes,
                                              num_levels=num_levels,
                                              regularizer_factor=regularizer_factor,
                                              channel_multiplier=channel_multiplier)
    lw_unet.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss=BinaryCrossentropy(name="weighted_binary_crossentropy"),
                    metrics=[CustomBinaryIoU(threshold=0.5, name="binary_IoU"),
                             BinaryAccuracy(threshold=0.5, name="binary_accuracy")])
    if pretrained_weight is not None:
        lw_unet.load_weights(filepath=pretrained_weight)
    return lw_unet


def _get_binary_contracting_block(inputs, conv2d_layer: [BinaryConv2D, BinarySeparableConv2D],
                                  num_activation_residual_levels, conv_residual_level_dict, filters, padding,
                                  kernel_initializer_seed, channel_multiplier=1, name="BinaryContractingBlock"):
    if conv2d_layer is BinaryConv2D:
        conv_name = "_BinaryConv2D"
    elif conv2d_layer is BinarySeparableConv2D:
        conv_name = "_BinarySeparableConv2D"
    else:
        conv_name = "UnknownConv2D"

    pool = layers.MaxPooling2D(pool_size=(2, 2), name=name + "_MaxPooling2D")(inputs)
    conv1 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_1",
                         channel_multiplier=channel_multiplier)(pool)
    conv1 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_1")(conv1)
    conv2 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_2",
                         channel_multiplier=channel_multiplier)(conv1)
    conv2 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_2")(conv2)
    return conv2


def _get_binary_expanding_block(inputs, skip_connection, conv2d_layer: [BinaryConv2D, BinarySeparableConv2D],
                                num_activation_residual_levels, conv_residual_level_dict, filters, padding,
                                kernel_initializer_seed, channel_multiplier=1, name="BinaryExpandingBlock"):
    if conv2d_layer is BinaryConv2D:
        conv_name = "_BinaryConv2D"
    elif conv2d_layer is BinarySeparableConv2D:
        conv_name = "_BinarySeparableConv2D"
    else:
        conv_name = "UnknownConv2D"
    up = layers.UpSampling2D(size=(2, 2), name=name + "_UpSampling2D")(inputs)
    conv1 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=2, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_1",
                         channel_multiplier=channel_multiplier)(up)
    conv1 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_1")(conv1)
    merge = layers.Concatenate(axis=3, name=name + "_Concatenate")([skip_connection, conv1])
    conv2 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_2",
                         channel_multiplier=channel_multiplier)(merge)
    conv2 = BinarySignActivation(num_residual_levels=num_activation_residual_levels,
                                 name=name + "_BinarySignActivation_2")(conv2)
    conv3 = conv2d_layer(**conv_residual_level_dict, filters=filters, kernel_size=3, padding=padding,
                         initializer_seed=kernel_initializer_seed, name=name + conv_name + "_3",
                         channel_multiplier=channel_multiplier)(conv2)
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
                                           channel_multiplier=1,
                                           output_classes=1,
                                           num_levels=5,
                                           conv_kernel_initializer_seed=1):
    separable_conv_arg_dict = {"channel_multiplier": channel_multiplier}
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
                name="Level{}_BinaryLightweight_Contracting".format(level),
                channel_multiplier=channel_multiplier
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
            name="Level{}_BinaryLightweight_Expanding".format(level),
            channel_multiplier=channel_multiplier
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
                                         channel_multiplier=1,
                                         pretrained_weight=None):
    binary_lightweight_unet = get_uncompiled_binary_lightweight_unet(
        input_size=input_size,
        num_activation_residual_levels=num_activation_residual_levels,
        num_conv_residual_levels=num_conv_residual_levels,
        num_residual_levels_depthwise_filter=num_depthwise_conv_residual_levels,
        num_residual_levels_pointwise_filter=num_pointwise_conv_residual_levels,
        output_classes=output_classes,
        num_levels=num_levels,
        conv_kernel_initializer_seed=initializer_seed,
        channel_multiplier=channel_multiplier
    )

    binary_lightweight_unet.compile(optimizer=Adam(learning_rate=learning_rate),
                                    loss=BinaryCrossentropy(name="binary_crossentropy"),
                                    metrics=[CustomBinaryIoU(threshold=0.5, name="binary_IoU"),
                                             BinaryAccuracy(threshold=0.5, name="binary_accuracy")])

    if pretrained_weight is not None:
        binary_lightweight_unet.load_weights(pretrained_weight)

    return binary_lightweight_unet


def get_teacher_vanilla_unet(input_size, trained_weights,
                             features_to_extract=(2, 5, 8, 11, 14, 19, 24, 29, 34, 35)):
    vanilla_unet = get_compiled_unet(input_size, pretrained_weights=trained_weights)

    teacher_outputs = []
    for layer_index, layer in enumerate(vanilla_unet.layers):
        if layer_index in features_to_extract:
            teacher_outputs.append(layer.output)

    teacher_vanilla_unet = Model(inputs=vanilla_unet.inputs, outputs=teacher_outputs)
    teacher_vanilla_unet.load_weights(filepath=trained_weights, by_name=True)
    return teacher_vanilla_unet


def get_student_lightweight_unet(input_size,
                                 features_to_extract=(2, 5, 8, 11, 14, 19, 24, 29, 34, 35)):
    lw_unet = get_compiled_lightweight_unet(input_size=input_size)
    student_outputs = []
    for layer_index, layer in enumerate(lw_unet.layers):
        if layer_index in features_to_extract:
            student_outputs.append(layer.output)
    student_lw_unet = Model(inputs=lw_unet.inputs, outputs=student_outputs)
    return student_lw_unet


if __name__ == '__main__':
    # *: tensorboard --logdir="E:\ED_MS\Semester_3\Codes\MyProject\tensorboard_logs"
    seed = None
    batch_size = 1
    target_size = (512, 512)
    weight_path = "./checkpoints/trained_weights/unet_agarpads_seg_evaluation2.hdf5"

    image_dir = "../../Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "../../Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "../../Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset = "DIC"

    trained_unet_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                               "2022-08-06_fine_tune_vanilla_unet_with_l2-fold_0/" \
                               "vanilla_unet-fine-tuned_IoU=0.8880.h5"

    trained_lw_unet_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/knowledge_distillation-lw_unet-fold_1.h5"
    from data import _load_an_image_np, _resize_with_pad_or_random_crop_and_rescale

    image = _load_an_image_np(image_path="E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated/img_000001_1.tif")
    image, _, _ = _resize_with_pad_or_random_crop_and_rescale(image, None, None, (512, 512))
    image = np.expand_dims(image, axis=[0, -1])

    unet = get_compiled_unet((*target_size, 1))
    # binary_unet = get_compiled_binary_unet((*target_size, 1))
    lw_unet = get_compiled_lightweight_unet((*target_size, 1), channel_multiplier=2)
    # blu_net = get_compiled_binary_lightweight_unet((*target_size, 1), channel_multiplier=2)

    for i, layer in enumerate(unet.layers):
        print(i, layer.name)
    print("=" * 30)
    for i, layer in enumerate(lw_unet.layers):
        print(i, layer.name)

    # pred_mask_1 = unet(image, training=False)
    # pred_mask_2 = binary_unet(image, training=False)
    # pred_mask_3 = lw_unet(image, training=False)
    # pred_mask_4 = blu_net(image, training=False)
