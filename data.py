import numpy as np
from scipy import interpolate
import tensorflow as tf


def illumination_voodoo(image, num_control_points=5):
    control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0, image.shape[0] - 1, image.shape[0]))

    weights = tf.convert_to_tensor(np.reshape(np.tile(np.reshape(curve, (curve.shape[0], 1)),
                                                      (1, image.shape[1])),
                                              image.shape), dtype=tf.float32)
    new_image = tf.math.multiply(image, weights)
    k = (tf.math.reduce_max(image) - tf.math.reduce_min(image)) / (
            tf.math.reduce_max(new_image) - tf.math.reduce_min(new_image))
    new_image = (new_image - tf.math.reduce_min(new_image)) * k + tf.math.reduce_min(image)

    return new_image
