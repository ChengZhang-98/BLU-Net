import numpy as np
from scipy import interpolate
import tensorflow as tf


# import elasticdeform.tf as etf


# def illumination_voodoo_tf(image, num_control_points=5):
#     shape = image.shape
#     image = image.numpy().squeeze()
#
#     control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
#     random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
#     mapping = interpolate.PchipInterpolator(control_points, random_points)
#     curve = mapping(np.linspace(0, image.shape[0] - 1, image.shape[0]))
#
#     new_image = np.multiply(
#         image,
#         np.reshape(
#             np.tile(np.reshape(curve, (curve.shape[0], 1)), (1, image.shape[1])),
#             image.shape,
#         ),
#     )
#     # Rescale values to original range:
#     new_image = np.interp(
#         new_image, (new_image.min(), new_image.max()), (image.min(), image.max())
#     )
#
#     new_image = tf.reshape(tf.convert_to_tensor(image, dtype=tf.float32), shape=shape)
#
#     return new_image
#
#
# def histogram_voodoo_tf(image, num_control_points=3):
#     """
#     This function kindly provided by Daniel Eaton from the Paulsson lab.
#     It performs an elastic deformation on the image histogram to simulate
#     changes in illumination
#     :param image:
#     :param num_control_points:
#     :return:
#     """
#     shape = image.shape
#     image = image.numpy().squeeze()
#
#     control_points = np.linspace(0, 1, num=num_control_points + 2)
#     sorted_points = np.copy(control_points)
#     random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
#     sorted_points[1:-1] = np.sort(random_points)
#     mapping = interpolate.PchipInterpolator(control_points, sorted_points)
#     new_image = mapping(image)
#
#     new_image = tf.reshape(tf.convert_to_tensor(image, dtype=tf.float32), shape=shape)
#     return new_image
#
#
# def elastic_deformation_tf():
#     raise RuntimeError("Not implemented yet")
