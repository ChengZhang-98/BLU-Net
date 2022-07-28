import glob
import operator
import os
import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from keras.utils import Sequence
from skimage import morphology as morph
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def _get_matched_data_df(image_dir, image_type, mask_dir, mask_type, weight_map_dir, weight_map_type, dataset):
    image_df = pd.DataFrame({"image": glob.glob(os.path.join(image_dir, "*." + image_type.lower()))})
    image_df.loc[:, "id"] = image_df.loc[:, "image"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])

    if mask_dir is None:
        image_df.loc[:, "mask"] = None
        matched_df = image_df.copy()
    else:
        mask_df = pd.DataFrame({"mask": glob.glob(os.path.join(mask_dir, "*." + mask_type.lower()))})
        if dataset.lower() == "training_2d":
            mask_df.loc[:, "id"] = mask_df.loc[:, "mask"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
        else:
            mask_df.loc[:, "id"] = mask_df.loc[:, "mask"].apply(lambda x: os.path.splitext(os.path.basename(x))[0][:-5])

        matched_df = pd.merge(image_df, mask_df, how="inner", on=["id"])

    if weight_map_dir is None:
        matched_df.loc[:, "weight_map"] = None
    else:
        weight_map_df = pd.DataFrame(
            {"weight_map": glob.glob(os.path.join(weight_map_dir, "*." + weight_map_type.lower()))})
        if dataset.lower() == "training_2d":
            weight_map_df.loc[:, "id"] = weight_map_df.loc[:, "weight_map"].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0]
            )
        else:
            weight_map_df.loc[:, "id"] = weight_map_df.loc[:, "weight_map"].apply(
                lambda x: os.path.splitext(os.path.basename(x))[0][:-8]
            )
        matched_df = pd.merge(matched_df, weight_map_df, how="inner", on=["id"])
    matched_df = matched_df.sort_values(by=["image"], ignore_index=True)
    return matched_df


def _load_an_image_np(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image.squeeze().astype(dtype=np.float32)


def _center_crop_np(image, mask, weight_map, target_size):
    start_tuple = tuple(map(lambda length_i, target_length_i: length_i // 2 - target_length_i // 2,
                            image.shape,
                            target_size))
    end_tuple = tuple(map(operator.add, start_tuple, target_size))
    slice_tuple = tuple(map(slice, start_tuple, end_tuple))

    new_image = image[slice_tuple[0], slice_tuple[1]]
    new_mask, new_weight_map = None, None
    if mask is not None:
        new_mask = mask[slice_tuple[0], slice_tuple[1]]
    if weight_map is not None:
        new_weight_map = weight_map[slice_tuple[0], slice_tuple[1]]
    return new_image, new_mask, new_weight_map


def _random_crop_np(image, mask, weight_map, target_size):
    margin_tuple = tuple(map(lambda length_i, target_length_i: length_i - target_length_i + 1,
                             image.shape[0:2],
                             target_size))
    start_tuple = tuple(map(np.random.randint, (0, 0), margin_tuple))
    end_tuple = tuple(map(operator.add, start_tuple, target_size))
    slice_tuple = tuple(map(slice, start_tuple, end_tuple))

    new_image = image[slice_tuple[0], slice_tuple[1]]
    new_mask, new_weight_map = None, None
    if mask is not None:
        new_mask = mask[slice_tuple[0], slice_tuple[1]]
    if weight_map is not None:
        new_weight_map = weight_map[slice_tuple[0], slice_tuple[1]]
    return new_image, new_mask, new_weight_map


def _center_pad_np(image, mask, weight_map, target_size, constant=0):
    start_tuple = tuple(map(lambda target_length_i, length_i: target_length_i // 2 - length_i // 2,
                            target_size,
                            image.shape))
    end_tuple = tuple(map(operator.add, start_tuple, image.shape))
    slice_tuple = tuple(map(slice, start_tuple, end_tuple))

    new_image = np.ones(target_size, dtype=image.dtype) * constant
    new_image[slice_tuple[0], slice_tuple[1]] = image
    new_mask, new_weight_map = None, None
    if mask is not None:
        new_mask = np.ones(target_size, dtype=mask.dtype) * constant
        new_mask[slice_tuple[0], slice_tuple[1]] = mask
    if weight_map is not None:
        new_weight_map = np.ones(target_size, dtype=weight_map.dtype) * constant
        new_weight_map[slice_tuple[0], slice_tuple[1]] = weight_map
    return new_image, new_mask, new_weight_map


def _resize_with_pad_or_random_crop_and_rescale(image, mask, weight_map, target_size, padding_constant=0):
    if image.shape[0] > target_size[0] and image.shape[1] > target_size[1]:
        new_image, new_mask, new_weight_map = _random_crop_np(image, mask, weight_map, target_size)
    elif image.shape[0] < target_size[0] and image.shape[1] < target_size[1]:
        new_image, new_mask, new_weight_map = _center_pad_np(image, mask, weight_map, target_size, padding_constant)
    else:
        padding_size = (max(target_size), max(target_size))
        new_image, new_mask, new_weight_map = _center_pad_np(image, mask, weight_map, padding_size, padding_constant)
        new_image, new_mask, new_weight_map = _random_crop_np(image, mask, weight_map, target_size)
    return new_image / 255.0, new_mask / 255.0, new_weight_map


def _resize_with_pad_or_center_crop_and_rescale(image, mask, weight_map, target_size, padding_constant=0):
    if image.shape[0] > target_size[0] and image.shape[1] > target_size[1]:
        new_image, new_mask, new_weight_map = _center_crop_np(image, mask, weight_map, target_size)
    elif image.shape[0] < target_size[0] and image.shape[1] < target_size[1]:
        new_image, new_mask, new_weight_map = _center_pad_np(image, mask, weight_map, target_size, padding_constant)
    else:
        padding_size = (max(target_size), max(target_size))
        new_image, new_mask, new_weight_map = _center_pad_np(image, mask, weight_map, padding_size, padding_constant)
        new_image, new_mask, new_weight_map = _center_crop_np(image, mask, weight_map, target_size)
    return new_image / 255.0, new_mask / 255.0, new_weight_map


def _reduce_size_before_resize(image, mask, weight_map, target_size):
    if image.shape[0] > 1.5 * target_size[0] or image.shape[1] > 1.5 * target_size[1]:
        # *: note that opencv2.resize requires parameter `dim` in w x h instead of h x w
        target_dim = (int(target_size[1] * 1.5), int(target_size[0] * 1.5))
        image = cv2.resize(image, target_dim, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, target_dim, interpolation=cv2.INTER_LINEAR)
        if weight_map is not None:
            weight_map = cv2.resize(weight_map, target_dim, interpolation=cv2.INTER_LINEAR)
    return image, mask, weight_map


class DataGenerator(Sequence):
    def __init__(self, batch_size, dataset_name, mode, use_weight_map,
                 image_dir,
                 image_type,
                 mask_dir,
                 mask_type,
                 weight_map_dir,
                 weight_map_type,
                 target_size,
                 data_aug_transform,
                 seed):
        super(DataGenerator, self).__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.mode = mode.lower()
        self.use_weight_map = use_weight_map
        self.target_size = target_size
        self.data_aug_transform = data_aug_transform
        self.seed = seed

        assert mode.lower() in ["train", "validate",
                                "test", "inference"], \
            "Unsupported mode: {}. mode should be 'train'/'validate'/'test'/'inference'".format(mode)

        if image_dir:
            self.data_df = _get_matched_data_df(image_dir, image_type, mask_dir, mask_type, weight_map_dir,
                                                weight_map_type, dataset_name)
            if self.seed is None:
                self.seed_gen = itertools.count(start=0, step=1)
            else:
                self.seed_gen = itertools.count(start=self.seed, step=1)
                self.data_df = self.data_df.sample(frac=1, random_state=self.seed, ignore_index=True)

    @classmethod
    def build_from_dataframe(cls, dataframe, batch_size, mode, use_weight_map, target_size, data_aug_transform, seed):
        data_gen = cls(batch_size, dataset_name=None, mode=mode, use_weight_map=use_weight_map,
                       image_dir=None, image_type=None,
                       mask_dir=None, mask_type=None,
                       weight_map_dir=None, weight_map_type=None,
                       target_size=target_size, data_aug_transform=data_aug_transform, seed=seed)
        data_gen.data_df = dataframe.reset_index(drop=True)

        if seed is None:
            data_gen.seed_gen = itertools.count(start=0, step=1)
        else:
            data_gen.seed_gen = itertools.count(start=seed, step=1)
            data_gen.data_df = dataframe.sample(frac=1, random_state=seed, ignore_index=True)

        return data_gen

    def __getitem__(self, index):
        """

        :param index:
        :return: image_batch, mask_batch
                 image_batch: BxHxWx1 tf.float32 tensor
                 mask_match: BxHxWx1 tf.float32 tensor
        """
        batch_df = self.data_df.iloc[self.batch_size * index: self.batch_size * (index + 1), :]
        image_batch = []
        mask_batch = []
        weight_map_batch = []

        for i, row in enumerate(batch_df.itertuples()):
            # load an image
            image_i = _load_an_image_np(row.image)
            # load the corresponding mask and weight map
            if self.mode != "inference":
                mask_i = _load_an_image_np(row.mask)
                if self.use_weight_map:
                    if self.dataset_name == "training_2d":
                        weight_map_i = _load_an_image_np(row.weight_map)
                    else:
                        weight_map_i = np.load(row.weight_map)
                else:
                    weight_map_i = None
            else:
                mask_i = None
                weight_map_i = None

            # data preprocessing
            # *: the image size in training_2d differs across a large range from ~400 to 2048
            # *: therefore it's necessary to reduce the size
            if self.dataset_name == "training_2d":
                image_i, mask_i, weight_map_i = _reduce_size_before_resize(image_i, mask_i, weight_map_i,
                                                                           target_size)
            if self.mode == "train":
                image_i, mask_i, weight_map_i = \
                    _resize_with_pad_or_random_crop_and_rescale(image=image_i, mask=mask_i, weight_map=weight_map_i,
                                                                target_size=self.target_size, padding_constant=0)
            else:
                image_i, mask_i, weight_map_i = \
                    _resize_with_pad_or_center_crop_and_rescale(image=image_i, mask=mask_i, weight_map=weight_map_i,
                                                                target_size=self.target_size, padding_constant=0)

            if self.data_aug_transform is not None and self.mode == "train":
                image_i, mask_i, weight_map_i = self.data_aug_transform(image_i, mask_i, weight_map_i)

            # convert numpy.ndarray to tf.Tensor
            image_batch.append(tf.convert_to_tensor(np.expand_dims(image_i, axis=-1), dtype=tf.float32))
            if mask_i is not None:
                mask_i = tf.convert_to_tensor(np.expand_dims(mask_i, axis=-1), dtype=tf.float32)
            if weight_map_i is not None:
                weight_map_i = tf.convert_to_tensor(np.expand_dims(weight_map_i, axis=-1), dtype=tf.float32)
            mask_batch.append(mask_i)
            weight_map_batch.append(weight_map_i)

        image_batch = tf.stack(image_batch, axis=0)

        if self.mode != "inference":
            mask_batch = tf.stack(mask_batch, axis=0)
            if self.use_weight_map:
                weight_map_batch = tf.stack(weight_map_batch, axis=0)
                return image_batch, mask_batch, weight_map_batch
            else:
                return image_batch, mask_batch
        else:
            return image_batch

    def __len__(self):
        return int(len(self.data_df) / self.batch_size)

    def on_epoch_end(self):
        if self.mode == "train":
            self.data_df = self.data_df.sample(frac=1, random_state=next(self.seed_gen))

    def get_batch_dataframe(self, index):
        batch_df = self.data_df.iloc[self.batch_size * index:self.batch_size * (index + 1), :]
        return batch_df


def _get_kernel(n: int):
    """
    *: Function kernel from Delta 2.0
    :param n:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    return kernel


def _estimate_class_weights(mask_dir, sample_size=None):
    """
    *: Function "data.estimate_seg2D_class-weights" from Delta 2.0
    :param mask_dir: directory to masks
    :param sample_size: None or int. Take a sample size of training set to reduce computation time.
    :return: (class1_weight, class2_weight)
    """
    mask_name_arr = glob.glob(os.path.join(mask_dir, "*.tif")) + glob.glob(os.path.join(mask_dir, "*.png"))
    assert len(mask_name_arr) != 0, "Empty mask dir: {}".format(mask_dir)

    # Take a sample size of training set to reduce computation time.
    if sample_size:
        mask_name_arr = np.random.choice(mask_name_arr, sample_size)

    c1 = 0
    c2 = 0

    for mask_name in mask_name_arr:
        mask = cv2.imread(mask_name, 0) / 255
        border = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _get_kernel(20))
        border[mask > 0] = 0

        background = np.ones(mask.shape)
        background = background - (mask + border)

        mask_erode = cv2.erode(mask, _get_kernel(2))
        mask_dil = cv2.dilate(mask, _get_kernel(3))
        border_erode = (mask_dil < 1) * (border > 0) * 1

        skel = morph.skeletonize(mask_erode > 0) * 1
        skel_border = morph.skeletonize(border_erode > 0) * 1

        c1 = c1 + np.sum(np.sum(skel.astype(np.float64)))
        c2 = c2 + np.sum(np.sum(skel_border.astype(np.float64)))

    if c1 > c2:
        class1 = c2 / c1
        class2 = 1.0
    else:
        class1 = 1.0
        class2 = 1.0 * c1 / c2

    return class1, class2


def _get_seg_weights(mask, class_weights=(1, 1)):
    """
    *: Function "data.seg_weights_2D" from Delta 2.0

    Compute custom weight maps designed for bacterial images where borders are difficult to distinguish
    :param mask: HxW numpy.ndarray
    :param class_weights: class weight tuple from estimate_class_weights
    :return: weight_map: HxW numpy.ndarray
    """

    border = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _get_kernel(20))
    border[mask > 0] = 0

    mask_erode = cv2.erode(mask, _get_kernel(2))

    mask_skel = morph.skeletonize(mask_erode > 0)
    border_skel = morph.skeletonize(border > 0)

    s, border_dist = morph.medial_axis(border_skel < 1, return_distance=True)
    s, mask_dist = morph.medial_axis(mask_skel < 1, return_distance=True)

    border_gra = border * (class_weights[1]) / (border_dist + 1) ** 2
    mask_gra = mask / (mask_dist + 1) ** 2

    weight_map = np.zeros(mask.shape, dtype=np.float32)

    weight_map[mask_erode > 0] = mask_gra[mask_erode > 0]
    weight_map[border > 0] = border_gra[border > 0]

    weight_map[mask_skel > 0] = class_weights[0]
    weight_map[border_skel > 0] = class_weights[1]

    background = np.ones(mask.shape) - mask - border
    weight_map[((weight_map == 0) * (background < 1))] = 1 / 255

    return weight_map


def calculate_and_save_weight_maps(mask_dir, weight_map_dir, sample_size=None):
    """
    Calculate and save HxWx1 weight map
    :param mask_dir:
    :param weight_map_dir:
    :param sample_size:
    :return:
    """
    mask_path_list = glob.glob(os.path.join(mask_dir, "*.tif"))
    class_weight_1, class_weight_2 = _estimate_class_weights(mask_dir, sample_size=sample_size)
    for mask_path in tqdm(mask_path_list):
        weight_map_name = os.path.splitext(os.path.basename(mask_path))[0].replace("mask", "weights") + ".npy"
        weight_map_path = os.path.join(weight_map_dir, weight_map_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        weight_map = _get_seg_weights(mask, class_weights=(class_weight_1, class_weight_2))
        # weight_map = np.expand_dims(weight_map, axis=-1)

        np.save(weight_map_path, weight_map)


def do_statistic_on_image_size(image_dir, image_type):
    h_list = []
    w_list = []
    image_path_list = glob.glob(os.path.join(image_dir, "*." + image_type.lower()))
    assert len(image_path_list) != 0, "No {} files in this directory {}".format(image_type, image_dir)

    for image_path in tqdm(image_path_list):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h_list.append(image.shape[0])
        w_list.append(image.shape[1])

    statistic_df = pd.DataFrame(dict(image=image_path_list, height=h_list, width=w_list))
    print(statistic_df.describe())
    return statistic_df


def _binarize_mask(pred_mask):
    threshold = (np.amin(pred_mask) + np.amax(pred_mask)) / 2
    return np.array(pred_mask > threshold, dtype=bool)


def postprocess_a_mask(pred_mask, binarize_threshold=0.5, remove_min_size=20):
    """
    Receive HxW np.ndarray predicted mask
    Return HxW np.ndarray post-processed mask
    :param pred_mask:
    :param binarize_threshold:
    :param remove_min_size:
    :return:
    """
    pred_mask = (pred_mask >= binarize_threshold).astype(bool)
    pred_mask = morph.remove_small_objects(pred_mask, min_size=remove_min_size, connectivity=1)

    return pred_mask


def postprocess_a_mask_batch(pred_mask_batch, binarize_threshold=0.5, remove_min_size=20):
    binary_mask_list = []
    for i in range(pred_mask_batch.shape[0]):
        mask_i = pred_mask_batch[i, ...].numpy().squeeze()
        mask_i = postprocess_a_mask(mask_i, binarize_threshold=binarize_threshold, remove_min_size=remove_min_size)
        binary_mask_list.append(np.expand_dims(mask_i, axis=-1))
    return tf.stack(binary_mask_list, axis=0)


if __name__ == '__main__':
    # *: DIC dataset
    # image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    # image_type = "tif"
    # mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    # mask_type = "tif"
    # weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    # weight_map_type = "npy"
    # dataset = "DIC"

    # *: Training_2D dataset
    image_dir = "E:/ED_MS/Semester_3/Dataset/training_2D/training/segmentation_set/img"
    image_type = "png"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/training_2D/training/segmentation_set/seg"
    mask_type = "png"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/training_2D/training/segmentation_set/wei"
    weight_map_type = "png"
    dataset = "training_2d"

    target_size = (512, 512)

    bool_calculate_and_save_weight_maps = False
    bool_data_generator_test = True
    bool_do_statistic_on_image_sizes = False

    # Data preprocessing
    # calculate and save weight map files
    if bool_calculate_and_save_weight_maps:
        calculate_and_save_weight_maps(mask_dir=mask_dir, weight_map_dir=weight_map_dir, sample_size=None)

    if bool_data_generator_test:
        data_gen_1 = DataGenerator(batch_size=1, dataset_name=dataset, mode="train", use_weight_map=True,
                                   image_dir=image_dir, image_type=image_type,
                                   mask_dir=mask_dir, mask_type=mask_type,
                                   weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
                                   target_size=target_size, data_aug_transform=None, seed=None)

        for i in range(5):
            index = np.random.randint(0, len(data_gen_1.data_df))
            image, mask, weight_map = data_gen_1[index]
            print(image.shape, mask.shape, weight_map.shape)
            image = image.numpy().squeeze()
            mask = mask.numpy().squeeze()
            weight_map = weight_map.numpy().squeeze()
            f, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(image, cmap="gray", vmin=0, vmax=1)
            ax[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            ax[2].imshow(weight_map)
            f.show()

    if bool_do_statistic_on_image_sizes:
        """
        *: training_2d
                height        width
        count   307.000000   307.000000
        mean   1114.462541  1192.970684
        std     652.918787   612.352724
        min     420.000000   350.000000
        25%     520.000000   696.000000
        50%     760.000000   816.000000
        75%    2048.000000  2048.000000
        max    2048.000000  2048.000000
        """
        static_df = do_statistic_on_image_size(image_dir, image_type)
