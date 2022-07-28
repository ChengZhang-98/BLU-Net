import tempfile

import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from data_augmentation import *


# todo: pruning

def quantize_unet(pretrained_unet, data_gen_train, data_gen_val, retrain_epochs, q_aware_checkpoint_path):
    q_aware_unet = tfmot.quantization.keras.quantize_model(pretrained_unet)
    q_aware_unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                         loss=tf.keras.losses.BinaryCrossentropy(name="weighted_binary_crossentropy"),
                         metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
                                  tf.keras.metrics.BinaryIoU(name="binary_IoU", target_class_ids=[1], threshold=0.5)])
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=q_aware_checkpoint_path,
                                                                   save_weights_only=False,
                                                                   verbosde=1,
                                                                   monitor='val_binary_IoU',
                                                                   mode='max',
                                                                   save_best_only=True)
    history = q_aware_unet.fit(x=data_gen_train, epochs=retrain_epochs,
                               validation_data=data_gen_val, shuffle=False,
                               validation_freq=1, callbacks=model_checkpoint_callback,
                               workers=1, use_multiprocessing=False, verbose=1)
    # float tflite model
    float_converter = tf.lite.TFLiteConverter.from_keras_model(pretrained_unet)
    float_tflite_unet = float_converter.convert()

    # quantized tflite model
    quantization_converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_unet)
    quantization_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_unet = quantization_converter.convert()

    _, float_file = tempfile.mkstemp(".tflite")
    _, quant_file = tempfile.mkstemp(".tflite")

    with open(quant_file, "wb") as f:
        f.write(quantized_tflite_unet)
    with open(float_file, "wb") as f:
        f.write(float_tflite_unet)

    float_unet_size = os.path.getsize(float_file) / float(2 ** 20)
    quant_unet_size = os.path.getsize(quant_file) / float(2 ** 20)
    quantization_result = "Float unet size = {:.2f} MB".format(float_unet_size) + "\n" + \
                          "Quantized unet size = {:.2f} MB".format(quant_unet_size) + "\n" + \
                          "Compression rate = {:.2f}×".format(float_unet_size / quant_unet_size)

    print(quantization_result)

    log_df = pd.DataFrame(dict(epoch=history.epoch) | history.history)

    return log_df, quantization_result
