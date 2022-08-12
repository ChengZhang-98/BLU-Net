import keras.layers as layers
import tensorflow as tf
from keras import Model
from tqdm import tqdm

from model import get_compiled_binary_lightweight_unet, get_compiled_lightweight_unet, get_compiled_unet
from residual_binarization import BinarySignActivation, BinaryConv2D, BinarySeparableConv2D


def count_full_precision_model_weights(model: Model):
    parameter_cnt = 0
    for layer in tqdm(model.layers):
        layer_parameter_cnt = 0
        for weight in layer.trainable_weights:
            layer_parameter_cnt += tf.reduce_prod(weight.shape)
        parameter_cnt += layer_parameter_cnt

    return parameter_cnt


# todo list compute the model size
def count_residual_binarized_model_weights(model: Model):
    fp32_parameter_cnt = 0
    bool_parameter_cnt = 0

    for layer in tqdm(model.layers):
        layer_fp32_parameter_cnt = 0
        layer_bool_parameter_cnt = 0

        if isinstance(layer, BinarySignActivation):
            fp32_parameter_cnt += tf.reduce_prod(layer.gamma.shape)
        elif isinstance(layer, BinaryConv2D):
            fp32_parameter_cnt += tf.reduce_prod(layer.gamma.shape)
            bool_parameter_cnt += tf.reduce_prod(layer.kernel.shape) * tf.reduce_prod(layer.gamma.shape)
            fp32_parameter_cnt += tf.reduce_prod(layer.bias.shape)
        elif isinstance(layer, BinarySeparableConv2D):
            fp32_parameter_cnt += tf.reduce_prod(layer.gamma_depthwise_kernel.shape)
            fp32_parameter_cnt += tf.reduce_prod(layer.gamma_pointwise_kernel.shape)
            bool_parameter_cnt += tf.reduce_prod(layer.gamma_depthwise_kernel.shape) * tf.reduce_prod(
                layer.pointwise_kernel.shape)
            bool_parameter_cnt += tf.reduce_prod(layer.gamma_pointwise_kernel.shape) * tf.reduce_prod(
                layer.depthwise_kernel.shape)
            fp32_parameter_cnt += tf.reduce_prod(layer.bias.shape)
        elif isinstance(layer, (layers.InputLayer, layers.MaxPool2D, layers.UpSampling2D,
                                layers.Concatenate, layers.Activation)):
            pass
        else:
            raise RuntimeError("Unsupported binary layer type: {}".format(type(layer)))

    return bool_parameter_cnt, fp32_parameter_cnt


def compute_model_size_in_mib(fp32_cnt=0, bool_cnt=0):
    return (32 * fp32_cnt + bool_cnt) / 2 ** 20 / 8


if __name__ == '__main__':
    # * results
    """
    U-Net: 118.37231826782227 MiB, 31030593 fp32 parameters

    Lightweight U-Net: 15.005619049072266 MiB, 3933633 fp32 parameters

    BLU-Net 1.4311981201171875 MiB, 11780352 binary parameters, 7044 fp32 parameters
    """
    input_size = (512, 512, 1)

    unet = get_compiled_unet(input_size)
    unet_param_cnt = count_full_precision_model_weights(unet)
    unet_size = compute_model_size_in_mib(fp32_cnt=unet_param_cnt)
    print("U-Net: {} MiB, {} fp32 parameters".format(unet_size, unet_param_cnt))

    lw_unet = get_compiled_lightweight_unet(input_size, channel_multiplier=1)
    lw_unet_param_cnt = count_full_precision_model_weights(lw_unet)
    lw_unet_size = compute_model_size_in_mib(fp32_cnt=lw_unet_param_cnt)
    print("Lightweight U-Net: {} MiB, {} fp32 parameters".format(lw_unet_size, lw_unet_param_cnt))

    blu_net = get_compiled_binary_lightweight_unet(input_size,
                                                   num_activation_residual_levels=3,
                                                   num_conv_residual_levels=3,
                                                   num_depthwise_conv_residual_levels=3,
                                                   num_pointwise_conv_residual_levels=3,
                                                   channel_multiplier=1)
    bool_param_cnt, fp32_param_cnt = count_residual_binarized_model_weights(blu_net)
    blu_net_size = compute_model_size_in_mib(fp32_cnt=fp32_param_cnt, bool_cnt=bool_param_cnt)
    print("BLU-Net {} MiB, {} binary parameters, {} fp32 parameters".format(blu_net_size,
                                                                            bool_param_cnt, fp32_param_cnt))
