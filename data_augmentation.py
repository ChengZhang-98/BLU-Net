import copy
import os
import time
from functools import partial

import cv2
import elasticdeform
import numpy as np
from scipy import interpolate
from skimage import transform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def illumination_voodoo_np(image, mask, weight_map, num_control_points=5):
    control_points = np.linspace(0, image.shape[0] - 1, num=num_control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    mapping = interpolate.PchipInterpolator(control_points, random_points)
    curve = mapping(np.linspace(0, image.shape[0] - 1, image.shape[0]))
    new_image = np.multiply(image,
                            np.reshape(np.tile(np.reshape(curve, curve.shape + (1,)), (1, image.shape[1])),
                                       image.shape))
    new_image = np.interp(new_image, (new_image.min(), new_image.max()), (image.min(), image.max()))

    return new_image, mask, weight_map


def histogram_voodoo(image, mask, weight_map, num_control_points=3):
    control_points = np.linspace(0, 1, num=num_control_points + 2)
    sorted_points = copy.copy(control_points)
    random_points = np.random.uniform(low=0.1, high=0.9, size=num_control_points)
    sorted_points[1:-1] = np.sort(random_points)
    mapping = interpolate.PchipInterpolator(control_points, sorted_points)
    new_image = mapping(image)
    new_image = np.interp(new_image, (new_image.min(), new_image.max()), (image.min(), image.max()))

    return new_image, mask, weight_map


def elasticdeform_np(image, mask, weight_map, sigma=10, points=3, mode="mirror", **kwargs):
    matrix_to_deform = [image, mask, weight_map]

    new_image, new_mask, new_weight_map = elasticdeform.deform_random_grid(matrix_to_deform, sigma=sigma,
                                                                           points=points, order=0,
                                                                           mode=mode, axis=(0, 1),
                                                                           prefilter=False, **kwargs)
    return new_image, new_mask, new_weight_map


def gaussian_noise_np(image, mask, weight_map, sigma=0.001, clip_value_min=0, clip_value_max=1):
    noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    new_image = np.clip(image + noise, a_min=clip_value_min, a_max=clip_value_max)
    return new_image, mask, weight_map


def gaussian_blur_np(image, mask, weight_map, filter_shape=(3, 3), sigma=1.0):
    new_image = cv2.GaussianBlur(image, filter_shape, sigma)
    return new_image, mask, weight_map


def random_flip_np(image, mask, weight_map):
    if np.random.randint(0, 2):
        image = np.fliplr(image)
        mask = np.fliplr(mask)
        weight_map = np.fliplr(weight_map)
    if np.random.randint(0, 2):
        image = np.flipud(image)
        mask = np.flipud(mask)
        weight_map = np.flipud(weight_map)
    return image, mask, weight_map


def random_rotate_np(image, mask, weight_map, max_angle=0.1, fill_mode="reflect",
                     interpolation_order=1):
    angle = np.random.uniform(-max_angle, max_angle)
    new_image = transform.rotate(image, angle, mode=fill_mode, order=interpolation_order)
    new_mask = transform.rotate(mask, angle, mode=fill_mode, order=interpolation_order)
    new_weight_map = transform.rotate(weight_map, angle, mode=fill_mode, order=interpolation_order)
    return new_image, new_mask, new_weight_map


def random_rot90(image, mask, weight_map):
    k = np.random.randint(0, 4)
    new_image = np.rot90(image, k=k)
    new_mask = np.rot90(mask, k=k)
    new_weight_map = np.rot90(weight_map, k=k)
    return new_image, new_mask, new_weight_map


def _shift(image, vector, order=1):
    affine_transform = transform.AffineTransform(translation=vector)
    shifted = transform.warp(image, affine_transform, mode="edge", order=order)

    return shifted


def _zoom_and_shift(image, zoom_level: float, shift_x: float, shift_y: float, order=1):
    old_shape = image.shape
    image = transform.rescale(image, zoom_level, mode="edge", order=order)
    shift_x = shift_x * image.shape[0]
    shift_y = shift_y * image.shape[1]
    image = _shift(image, (shift_y, shift_x), order=order)
    i0 = (
        round(image.shape[0] / 2 - old_shape[0] / 2),
        round(image.shape[1] / 2 - old_shape[1] / 2),
    )
    image = image[i0[0]: (i0[0] + old_shape[0]), i0[1]: (i0[1] + old_shape[1])]
    return image


def random_zoom_and_shift_np(image, mask, weight_map, zoom_beta=0.05, shift_x_max=0.05, shift_y_max=0.05):
    zoom = np.random.exponential(zoom_beta)
    shift_x = np.random.uniform(-shift_x_max, shift_y_max)
    shift_y = np.random.uniform(-shift_y_max, shift_y_max)

    new_image = _zoom_and_shift(image, zoom_level=zoom + 1, shift_x=shift_x, shift_y=shift_y)
    new_mask = _zoom_and_shift(mask, zoom_level=zoom + 1, shift_x=shift_x, shift_y=shift_y)
    new_weight_map = _zoom_and_shift(weight_map, zoom_level=zoom + 1, shift_x=shift_x, shift_y=shift_y)

    return new_image, new_mask, new_weight_map


class _DataAugBase:
    def __init__(self, aug_func):
        self.aug_func = aug_func
        self.kwargs = None

    def __call__(self, image, mask, weight_map):
        if self.kwargs is None:
            return self.aug_func(image, mask, weight_map)
        else:
            return self.aug_func(image, mask, weight_map, **self.kwargs)


class IlluminationVoodoo(_DataAugBase):
    def __init__(self, num_control_points=3):
        super(IlluminationVoodoo, self).__init__(illumination_voodoo_np)
        self.kwargs = {"num_control_points": num_control_points}


class HistogramVoodoo(_DataAugBase):
    def __init__(self, num_control_points=3):
        super(HistogramVoodoo, self).__init__(histogram_voodoo)
        self.kwargs = {"num_control_points": num_control_points}


class ElasticDeform(_DataAugBase):
    def __init__(self, sigma=10, points=3, mode="mirror", **kwargs):
        super(ElasticDeform, self).__init__(elasticdeform_np)
        self.kwargs = {"sigma": sigma, "points": points, "mode": mode} | kwargs


class GaussianNoise(_DataAugBase):
    def __init__(self, sigma=1 / 255, clip_value_min=0, clip_value_max=1):
        super(GaussianNoise, self).__init__(gaussian_noise_np)
        self.kwargs = {"sigma": sigma, "clip_value_min": clip_value_min, "clip_value_max": clip_value_max}


class GaussianBlur(_DataAugBase):
    def __init__(self, filter_shape=(3, 3), sigma=1.0):
        super(GaussianBlur, self).__init__(gaussian_blur_np)
        self.kwargs = {"filter_shape": filter_shape, "sigma": sigma}


class RandomFlip(_DataAugBase):
    def __init__(self):
        super(RandomFlip, self).__init__(random_flip_np)


class RandomRotate(_DataAugBase):
    def __init__(self, max_angle=0.1, fill_mode="reflect", interpolation_order=1):
        super(RandomRotate, self).__init__(random_rotate_np)
        self.kwargs = {"max_angle": max_angle, "fill_mode": fill_mode, "interpolation_order": interpolation_order}


class RandomRot90(_DataAugBase):
    def __init__(self):
        super(RandomRot90, self).__init__(random_rot90)


class RandomZoomAndShift(_DataAugBase):
    def __init__(self, zoom_beta=0.05, shift_x_max=0.05, shift_y_max=0.05):
        super(RandomZoomAndShift, self).__init__(random_zoom_and_shift_np)
        self.kwargs = {"zoom_beta": zoom_beta, "shift_x_max": shift_x_max, "shift_y_max": shift_y_max}


class DataAugmentation:
    def __init__(self, augmentation_func_list=None):
        self.augmentation_func_list = augmentation_func_list

    def __call__(self, image, mask, weight_map):
        if self.augmentation_func_list is None:
            return image, mask, weight_map
        else:
            for func in self.augmentation_func_list:
                image, mask, weight_map = func(image, mask, weight_map)
            return image, mask, weight_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated/img_000006_1.tif"
    mask_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks/img_000006_1_mask.tif"
    weight_map_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights/img_000006_1_weights.npy"
    imshow_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}


    def show_the_triple(image, mask, weight_map, title, imshow_kwargs=None):
        if imshow_kwargs is None:
            imshow_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}
        f, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
        axes[0].imshow(image, **imshow_kwargs)
        axes[1].imshow(mask, **imshow_kwargs)
        axes[2].imshow(weight_map)
        f.suptitle(title)
        f.show()


    from data import _load_an_image_np, _resize_with_pad_or_random_crop_and_rescale

    image = _load_an_image_np(image_path)
    mask = _load_an_image_np(mask_path)
    weight_map = np.load(weight_map_path)
    show_the_triple(image, mask, weight_map, "Original", {"cmap": "gray", "vmin": 0, "vmax": 255})

    image, mask, weight_map = _resize_with_pad_or_random_crop_and_rescale(image,
                                                                          mask,
                                                                          weight_map,
                                                                          target_size=(256, 256))
    show_the_triple(image, mask, weight_map, "Resized and Rescaled")

    data_augmentation_transform = DataAugmentation([HistogramVoodoo(),
                                                    ElasticDeform(sigma=20),
                                                    GaussianNoise(sigma=1 / 255),
                                                    RandomFlip(),
                                                    RandomRotate(),
                                                    RandomZoomAndShift()])

    time_list = []
    for i in range(10):
        start = time.time()
        image, mask, weight_map = data_augmentation_transform(image, mask, weight_map)
        time_list.append(time.time() - start)
    show_the_triple(image, mask, weight_map, "All Augmentation Methods")
    print(sum(time_list) / len(time_list))
