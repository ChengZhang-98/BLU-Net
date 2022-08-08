import time
import os

import tensorflow as tf

from model import get_compiled_unet, get_uncompiled_unet


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
    input_size = (512, 512, 1)

    # *: vanilla unet
    # U-Net computation throughput = 216.941212 M FLOPs
    # inference latency of unet = 0.0555 s, 18.01 FPS
    unet = get_uncompiled_unet(input_size)
    unet.compile()
    inference_latency = measure_inference_time_ns(input_size, unet, 128)
    flops = measure_flops(unet)
    print("inference latency of unet = {:.4f} s, {:.2f} FPS".format(inference_latency / 1e9, 1e9/inference_latency))
    print("U-Net computation throughput = {} M FLOPs".format(flops / 1e6))
