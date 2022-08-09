import time
import os

import tensorflow as tf

from model import get_uncompiled_unet, get_uncompiled_lightweight_unet, get_uncompiled_binary_lightweight_unet


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


def _get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops


def measure_flops(uncompiled_model):
    uncompiled_model.compile()
    temp_path = "./temp_model_for_measure_flops.h5"
    unet.save(filepath=temp_path)
    flops = _get_flops(temp_path)
    os.remove(temp_path)
    return flops


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
    flops_unet = measure_flops(unet)

    # *: lightweight unet
    lw_unet = get_uncompiled_lightweight_unet(input_size=input_size, channel_multiplier=1)
    lw_unet.compile()
    inference_latency_lw_unet = measure_inference_time_ns(input_size, lw_unet, 128)
    flops_lw_unet = measure_flops(lw_unet)

    print("inference latency of vanilla unet = {:.4f} s, {:.2f} FPS".format(inference_latency_unet / 1e9,
                                                                            1e9 / inference_latency_unet))
    print("vanilla unet computation throughput = {} M FLOPs".format(flops_unet / 1e6))

    print("inference latency of lw_unet = {:.4f} s, {:.2f} FPS".format(inference_latency_lw_unet / 1e9,
                                                                       1e9 / inference_latency_lw_unet))
    print("lw_unet computation throughput = {} M FLOPs".format(flops_lw_unet / 1e6))
