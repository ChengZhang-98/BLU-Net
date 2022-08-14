import keras
import numpy as np
import tensorflow as tf
from keras import layers
from tensorflow import nn
from tqdm import tqdm


def _binarize(x):
    """
    a trick of implementing straight-through estimator (hard tanh) in tensorflow
        - sign(x) for forward propagation
        - clip(x) for backward propagation
    for straight-through estimator, refer to https://arxiv.org/abs/1602.02830
    :param x:
    :return:
    """
    clipped = tf.clip_by_value(x, -1, 1)
    binarized = tf.sign(clipped)
    return clipped + tf.stop_gradient(binarized - clipped)


def get_initial_gamma_values(num_residual_levels):
    initial_gamma = np.arange(num_residual_levels) + 1.0
    initial_gamma = initial_gamma[::-1]
    initial_gamma = initial_gamma / np.sum(initial_gamma)
    return initial_gamma


def _approximate_weights_via_residual_binarization(w, gamma):
    residual = w
    approximated_w = tf.zeros_like(w)
    for gamma_m in gamma:
        temp = tf.abs(gamma_m) * _binarize(residual)
        residual = residual - temp
        approximated_w = approximated_w + temp
    return approximated_w


def get_binary_representation(w, gamma):
    gamma_b_pairs = []
    residual = tf.convert_to_tensor(w)
    approximate_w = tf.zeros_like(w, trainable=False)
    gamma = tf.convert_to_tensor(gamma)
    for gamma_m in gamma:
        b = tf.sign(residual)
        residual = residual - tf.abs(gamma_m) * b
        gamma_b_pairs.append((gamma_m.numpy(), tf.cast(b, dtype=tf.bool).numpy()))
    return gamma_b_pairs


class BinarySignActivation(layers.Layer):
    """
    refer to ReBNet (https://arxiv.org/abs/1711.01243)
    """

    def __init__(self, num_residual_levels, name=None):
        super(BinarySignActivation, self).__init__(name=name)
        self.num_residual_levels = num_residual_levels
        self.gamma = None

    def build(self, input_shape):
        initial_gamma = get_initial_gamma_values(self.num_residual_levels)
        constant_initializer = tf.constant_initializer(initial_gamma)
        self.gamma = self.add_weight(name="gamma", shape=initial_gamma.shape,
                                     dtype=tf.float32,
                                     initializer=constant_initializer,
                                     trainable=True)

    def call(self, inputs, *args, **kwargs):
        # residual = inputs  # variable r in Algorithm 1
        # approximated_outputs = tf.zeros_like(inputs)
        # for m in range(self.num_residual_levels):
        #     gamma_m = self.gamma[m]
        #     temp = tf.abs(gamma_m) * binarize(residual)
        #     residual = residual - temp
        #     approximated_outputs = approximated_outputs + temp
        # return approximated_outputs
        return _approximate_weights_via_residual_binarization(inputs, self.gamma)

    def get_config(self):
        return dict(num_residual_levels=self.num_residual_levels)

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BinaryConv2D(layers.Layer):
    def __init__(self, num_residual_levels, filters, kernel_size, strides=(1, 1), padding='VALID', name=None, **kwargs):
        super(BinaryConv2D, self).__init__(name=name)
        self.num_residual_levels = num_residual_levels
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.kwargs = kwargs

        self.kernel = None
        self.bias = None
        self.gamma = None

    def build(self, input_shape):
        kernel_initializer_seed = self.kwargs.get("initializer_seed", 1)
        self.kernel = self.add_weight(name="kernel",
                                      shape=(self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
                                      dtype=tf.float32,
                                      initializer=tf.initializers.HeNormal(kernel_initializer_seed),
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=[self.filters],
                                    dtype=tf.float32,
                                    initializer=tf.initializers.Zeros(),
                                    trainable=True)
        initial_gamma = get_initial_gamma_values(self.num_residual_levels)
        constant_initializer = tf.constant_initializer(initial_gamma)
        self.gamma = self.add_weight(name="gamma", shape=initial_gamma.shape,
                                     dtype=tf.float32,
                                     initializer=constant_initializer,
                                     trainable=True)

    def call(self, inputs, *args, **kwargs):
        approximated_kernel = _approximate_weights_via_residual_binarization(w=self.kernel, gamma=self.gamma)
        return nn.bias_add(nn.conv2d(inputs, filters=approximated_kernel, strides=self.strides, padding=self.padding),
                           self.bias)

    def get_config(self):
        return dict(num_residual_levels=self.num_residual_levels,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding) | self.kwargs

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BinarySeparableConv2D(layers.Layer):
    def __init__(self, num_residual_levels_depthwise_filter, num_residual_levels_pointwise_filter, filters, kernel_size,
                 strides=(1, 1), padding='VALID', name=None, channel_multiplier=1, **kwargs):
        super(BinarySeparableConv2D, self).__init__(name=name)
        self.num_residual_levels_depthwise_filter = num_residual_levels_depthwise_filter
        self.num_residual_levels_pointwise_filter = num_residual_levels_pointwise_filter
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = (1, *strides, 1)
        self.padding = padding.upper()
        self.kwargs = kwargs
        self.channel_multiplier = channel_multiplier

        self.depthwise_kernel = None
        self.pointwise_kernel = None
        self.bias = None
        self.gamma_depthwise_kernel = None
        self.gamma_pointwise_kernel = None

    def build(self, input_shape):
        kernel_initializer_seed = self.kwargs.get("initializer_seed", 1)
        self.depthwise_kernel = self.add_weight(name="depthwise_filter",
                                                shape=(self.kernel_size, self.kernel_size, input_shape[-1],
                                                       self.channel_multiplier),
                                                dtype=tf.float32,
                                                initializer=tf.initializers.HeNormal(kernel_initializer_seed),
                                                trainable=True)
        self.pointwise_kernel = self.add_weight(name="pointwise_filter",
                                                shape=(1, 1, self.channel_multiplier * input_shape[-1], self.filters),
                                                dtype=tf.float32,
                                                initializer=tf.initializers.HeNormal(kernel_initializer_seed),
                                                trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=self.filters,
                                    dtype=tf.float32,
                                    initializer=tf.initializers.Zeros(),
                                    trainable=True)
        initial_gamma_depthwise_kernel = get_initial_gamma_values(self.num_residual_levels_depthwise_filter)
        constant_initializer_depthwise_kernel = tf.constant_initializer(initial_gamma_depthwise_kernel)
        initial_gamma_pointwise_kernel = get_initial_gamma_values(self.num_residual_levels_pointwise_filter)
        constant_initializer_pointwise_kernel = tf.constant_initializer(initial_gamma_pointwise_kernel)
        self.gamma_depthwise_kernel = self.add_weight(name="gamma_depthwise_filter",
                                                      shape=initial_gamma_depthwise_kernel.shape,
                                                      dtype=tf.float32,
                                                      initializer=constant_initializer_depthwise_kernel,
                                                      trainable=True)
        self.gamma_pointwise_kernel = self.add_weight(name="gamma_pointwise_filter",
                                                      shape=initial_gamma_pointwise_kernel.shape,
                                                      dtype=tf.float32,
                                                      initializer=constant_initializer_pointwise_kernel,
                                                      trainable=True)

    def call(self, inputs, *args, **kwargs):
        approximated_w_depthwise_filter = _approximate_weights_via_residual_binarization(self.depthwise_kernel,
                                                                                         self.gamma_depthwise_kernel)
        approximated_w_pointwise_filter = _approximate_weights_via_residual_binarization(self.pointwise_kernel,
                                                                                         self.gamma_pointwise_kernel)
        return nn.bias_add(nn.separable_conv2d(inputs, depthwise_filter=approximated_w_depthwise_filter,
                                               pointwise_filter=approximated_w_pointwise_filter, strides=self.strides,
                                               padding=self.padding),
                           self.bias)

    def get_config(self):
        return dict(num_residual_levels_depthwise_filter=self.num_residual_levels_depthwise_filter,
                    num_residual_levels_pointwise_filter=self.num_residual_levels_pointwise_filter,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding) | self.kwargs

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def transfer_unet_weights_to_binary_unet(unet, binary_unet):
    # refer to keras source codes https://github.com/keras-team/keras/blob/v2.9.0/keras/engine/base_layer.py#L1570-L1654
    list_of_tf_param_np_weight_tuples = []
    for binary_unet_layer in tqdm(binary_unet.layers, ascii=True):
        binary_unet_layer_name = binary_unet_layer.name

        if "Conv" not in binary_unet_layer_name:
            continue

        link_to_target_kernel = binary_unet_layer.kernel
        link_to_target_bias = binary_unet_layer.bias
        corresponding_conv_layer_in_unet = binary_unet_layer_name.replace("Binary", "")

        layer_found = False
        for unet_layer in unet.layers:
            if unet_layer.name == corresponding_conv_layer_in_unet:
                np_source_kernel = unet_layer.kernel.numpy()
                np_source_bias = unet_layer.bias.numpy()
                assert link_to_target_kernel.shape == np_source_kernel.shape, \
                    "unmatched kernel size between {} and {}".format(binary_unet_layer.name, unet_layer.name)
                assert link_to_target_bias.shape == np_source_bias.shape, \
                    "unmatched bias size between {} and {}".format(binary_unet_layer.name, unet_layer.name)
                layer_found = True
                list_of_tf_param_np_weight_tuples.append((link_to_target_kernel, np_source_kernel))
                list_of_tf_param_np_weight_tuples.append((link_to_target_bias, np_source_bias))
        if not layer_found:
            raise RuntimeError("Failed to find matched layer for {}".format(binary_unet_layer.name))

    keras.backend.batch_set_value(list_of_tf_param_np_weight_tuples)
    return binary_unet


def transfer_lightweight_unet_weights_to_binary_lightweight_unet(lightweight_unet, binary_lightweight_unet):
    # note that the last conv layer is not transferred
    list_of_tf_param_np_weight_tuples = []
    for blw_unet_layer in tqdm(binary_lightweight_unet.layers):
        blw_unet_layer_name = blw_unet_layer.name

        if "Conv" not in blw_unet_layer_name:
            continue

        if "BinaryConv" in blw_unet_layer_name:
            link_to_target_kernel = blw_unet_layer.kernel
            link_to_target_bias = blw_unet_layer.bias
            corresponding_conv_layer_in_lw_unet = blw_unet_layer_name.replace("BinaryConv", "Conv")
        else:
            link_to_target_depthwise_kernel = blw_unet_layer.depthwise_kernel
            link_to_target_pointwise_kernel = blw_unet_layer.pointwise_kernel
            link_to_target_bias = blw_unet_layer.bias
            corresponding_conv_layer_in_lw_unet = \
                blw_unet_layer_name.replace("BinaryLightweight", "Lightweight").replace("BinarySeparableConv",
                                                                                        "SeparableConv")

        layer_found = False
        for lw_unet_layer in lightweight_unet.layers:
            if lw_unet_layer.name == corresponding_conv_layer_in_lw_unet:
                if "BinaryConv" in blw_unet_layer_name:
                    np_source_kernel = lw_unet_layer.kernel.numpy()
                    np_source_bias = lw_unet_layer.bias.numpy()
                    assert link_to_target_kernel.shape == np_source_kernel.shape, \
                        "unmatched kernel size between {} and {}".format(lw_unet_layer.name, blw_unet_layer.name)
                    assert link_to_target_bias.shape == np_source_bias.shape, \
                        "unmatched bias size between {} and {}".format(lw_unet_layer.name, blw_unet_layer.name)
                    layer_found = True
                    list_of_tf_param_np_weight_tuples.append((link_to_target_kernel, np_source_kernel))
                    list_of_tf_param_np_weight_tuples.append((link_to_target_bias, np_source_bias))
                else:
                    np_source_depthwise_kernel = lw_unet_layer.depthwise_kernel
                    np_source_pointwise_kernel = lw_unet_layer.pointwise_kernel
                    np_source_bias = lw_unet_layer.bias

                    assert link_to_target_depthwise_kernel.shape == np_source_depthwise_kernel.shape, \
                        "unmatched depthwise kernel size between {} and {}".format(blw_unet_layer.name,
                                                                                   lw_unet_layer.name)
                    assert link_to_target_pointwise_kernel.shape == np_source_pointwise_kernel.shape, \
                        "unmatched pointwise kernel size between {} and {}".format(blw_unet_layer.name,
                                                                                   lw_unet_layer.name)
                    assert link_to_target_bias.shape == np_source_bias.shape, \
                        "unmatched bias size between {} and {}".format(blw_unet_layer.name, lw_unet_layer.name)

                    layer_found = True
                    list_of_tf_param_np_weight_tuples.append(
                        (link_to_target_depthwise_kernel, np_source_depthwise_kernel))
                    list_of_tf_param_np_weight_tuples.append(
                        (link_to_target_pointwise_kernel, np_source_pointwise_kernel))
                    list_of_tf_param_np_weight_tuples.append((link_to_target_bias, np_source_bias))

        if not layer_found:
            raise RuntimeError("Failed to find matched layer for {}".format(blw_unet_layer.name))

    keras.backend.batch_set_value(list_of_tf_param_np_weight_tuples)
    return binary_lightweight_unet


def expand_blu_net_dimensions(blu_net, initial_weights):
    pass


if __name__ == '__main__':
    tf.keras.backend.clear_session()
    from model import get_compiled_lightweight_unet, get_compiled_binary_lightweight_unet

    lw_unet = get_compiled_lightweight_unet((512, 512, 1))
    blu_net = get_compiled_binary_lightweight_unet((512, 512, 1))

    for index, layer in enumerate(blu_net.layers):
        print(index, layer.name)
