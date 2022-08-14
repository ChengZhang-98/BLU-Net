import time

import tensorflow as tf

from model import get_uncompiled_unet, get_uncompiled_lightweight_unet


def measure_inference_time_ns(input_size, model, repeat=100):
    with tf.device("/GPU:0"):
        rand_input = tf.random.uniform(shape=(1, *input_size), dtype=tf.float32)

        for i in range(10):
            _ = model(rand_input, training=False)

        start = time.time_ns()
        for i in range(repeat):
            _ = model(rand_input, training=False)
        end = time.time_ns()
    return (end - start) / repeat


if __name__ == '__main__':
    # todo list
    # ! - [ ] the acceleration of the blu-net
    input_size = (512, 512, 1)

    # *: vanilla unet
    # U-Net computation throughput = 216.941212 M FLOPs
    # inference latency of unet = 0.0555 s, 18.01 FPS
    unet = get_uncompiled_unet(input_size)
    unet.compile()
    inference_latency_unet = measure_inference_time_ns(input_size, unet, 128)

    # *: lightweight unet
    lw_unet = get_uncompiled_lightweight_unet(input_size=input_size, channel_multiplier=1)
    lw_unet.compile()
    inference_latency_lw_unet = measure_inference_time_ns(input_size, lw_unet, 128)

    print("inference latency of vanilla unet = {:.4f} s, {:.2f} FPS".format(inference_latency_unet / 1e9,
                                                                            1e9 / inference_latency_unet))

    print("inference latency of lw_unet = {:.4f} s, {:.2f} FPS".format(inference_latency_lw_unet / 1e9,
                                                                       1e9 / inference_latency_lw_unet))
