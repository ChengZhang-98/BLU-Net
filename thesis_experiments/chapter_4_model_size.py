import tensorflow as tf
from keras import Model
from tqdm import tqdm

from model import get_compiled_unet, get_compiled_lightweight_unet, get_compiled_binary_lightweight_unet


def count_full_precision_model_weights(model: Model):
    parameter_cnt = 0
    for layer in tqdm(model.layers):
        layer_parameter_cnt = 0
        for weight in layer.trainable_weights:
            layer_parameter_cnt += tf.reduce_prod(weight.shape)
        parameter_cnt += layer_parameter_cnt

    return parameter_cnt


def count_residual_binarized_model_weights(model: Model):
    fp32_parameter_cnt = 0
    bool_parameter_cnt = 0

    for layer in tqdm(model.layers):
        layer_fp32_parameter_cnt = 0
        layer_bool_parameter_cnt = 0
        for weight in layer.trainable_weights:
            if "gamma" in weight.name.lower():
                layer_fp32_parameter_cnt += tf.reduce_prod(weight.shape)
            else:
                layer_bool_parameter_cnt += tf.reduce_prod(weight.shape)
        fp32_parameter_cnt += layer_fp32_parameter_cnt
        bool_parameter_cnt += layer_bool_parameter_cnt

    return bool_parameter_cnt, fp32_parameter_cnt


def compute_model_size_in_mib(fp32_cnt=0, bool_cnt=0):
    return (32 * fp32_cnt + bool_cnt) / 2 ** 20 / 8


if __name__ == '__main__':
    # * results
    """
    U-Net: 118.37231826782227 MiB, 31030593 fp32 parameters

    Lightweight U-Net: 15.005619049072266 MiB, 3933633 fp32 parameters

    BLU-Net 0.4696694612503052 MiB, 3933633 binary parameters, 195 fp32 parameters
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
