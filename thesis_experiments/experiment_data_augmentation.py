from functools import partial

import matplotlib.pyplot as plt

from data_augmentation import *
from data import _resize_with_pad_or_center_crop_and_rescale


def set_axis_invisible(ax):
    if isinstance(ax, (list, tuple)):
        for ax_i in ax:
            ax_i.xaxis.set_visible(False)
            ax_i.yaxis.set_visible(False)
    else:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


def show_the_triple(image, mask, weight_map, axes, axis_title_list=None):
    imshow_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1.0}
    abc = ["a", "b", "c"]
    axes[0].imshow(image, **imshow_kwargs)
    axes[1].imshow(mask, **imshow_kwargs)
    axes[2].imshow(weight_map)
    set_axis_invisible([axes[0], axes[1], axes[2]])

    if axis_title_list is not None:
        for ii, title_list in axis_title_list:
            axes[ii].set_title(axis_title_list[0], y=0, pad=-20, verticalalignment="top")
            axes[ii].set_title(axis_title_list[1], y=0, pad=-20, verticalalignment="top")
            axes[ii].set_title(axis_title_list[2], y=0, pad=-20, verticalalignment="top")
    else:
        axes[0].set_title("(a) Original photo", y=0, pad=-20, verticalalignment="top")
        axes[1].set_title("(b) Mask", y=0, pad=-20, verticalalignment="top")
        axes[2].set_title("(c) Weight Map", y=0, pad=-20, verticalalignment="top")


if __name__ == '__main__':
    image_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated/img_000006_1.tif"
    mask_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks/img_000006_1_mask.tif"
    weight_map_path = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights/img_000006_1_weights.npy"

    imshow_kwargs = dict(cmap="gray", vmin=0, vmax=1.0)
    letter_list = list(chr(i) for i in range(97, 97 + 26))
    axis_title_kwargs = dict(y=0, pad=-20, verticalalignment="top")
    saved_figure_dir = "E:/ED_MS/Semester_3/Codes/MyProject/saved_figures"
    conclusion_list = []

    # *: ----------------------------------
    show_illumination_voodoo_param_cmp = False
    show_histogram_voodoo_param_cmp = False
    show_elastic_deform_param_cmp = False
    show_gaussian_nose = True
    # *: ----------------------------------

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    weight_map = np.load(weight_map_path)

    image, mask, weight_map = _resize_with_pad_or_center_crop_and_rescale(image, mask, weight_map,
                                                                          target_size=(512, 512))
    f, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
    show_the_triple(image, mask, weight_map, axes)
    f.suptitle("Original Cell Photo, Mask, and Weight Map")
    f.show()
    f.savefig(os.path.join(saved_figure_dir, "original_triple.jpg"))

    if show_illumination_voodoo_param_cmp:
        num_control_points_list = [3, 4, 5]
        f, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)

        axes[0][0].imshow(image, **imshow_kwargs)
        axes[0][0].set_title("(a) original image", **axis_title_kwargs)
        set_axis_invisible(axes[0][0])

        for index, num_control_points in enumerate(num_control_points_list):
            index += 1
            row = index // 2
            column = index % 2
            image_iv, _, _ = illumination_voodoo_np(image, mask, weight_map, num_control_points)
            axes[row][column].imshow(image_iv, **imshow_kwargs)
            axes[row][column].set_title("({}) num_control_points = {}".format(letter_list[index], num_control_points),
                                        **axis_title_kwargs)
            set_axis_invisible(axes[row][column])
        f.suptitle("Illumination Voodoo", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "illumination_voodoo_param_cmp.jpg"))
        f.show()

        conclusion_list.append("For illumination_voodoo, "
                               "num_control_points = 3 is a sensible configuration")

    if show_histogram_voodoo_param_cmp:
        num_control_points_list = [2, 6, 10]
        f, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)

        axes[0][0].imshow(image, **imshow_kwargs)
        axes[0][0].set_title("(a) original image", **axis_title_kwargs)
        set_axis_invisible(axes[0][0])

        for index, num_control_points in enumerate(num_control_points_list):
            index += 1
            row = index // 2
            column = index % 2
            image_hv, _, _ = histogram_voodoo(image, mask, weight_map, num_control_points)
            axes[row][column].imshow(image_hv, **imshow_kwargs)
            axes[row][column].set_title("({}) num_control_points = {}".format(letter_list[index], num_control_points),
                                        **axis_title_kwargs)
            set_axis_invisible(axes[row][column])
        f.suptitle("Histogram Voodoo", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "histogram_voodoo_param_cmp.jpg"))
        f.show()

        conclusion_list.append("For histogram_voodoo, the value of num_control_points "
                               "does not make much difference on the output")

    if show_elastic_deform_param_cmp:
        sigma_list = [8, 10, 12]
        points_list = [2, 3, 4]

        f, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=200)

        for sigma_index, sigma in enumerate(sigma_list):
            for points_index, points in enumerate(points_list):
                image_ed, _, _ = elasticdeform_np(image, mask, weight_map, sigma=sigma, points=points,
                                                  mode="constant")
                axes[sigma_index][points_index].imshow(image_ed, **imshow_kwargs)
                axes[sigma_index][points_index].set_title(
                    "({}) sigma = {}, points = {}".format(letter_list[sigma_index * 3 + points_index],
                                                          sigma,
                                                          points),
                    **axis_title_kwargs)
                set_axis_invisible(axes[sigma_index][points_index])
        f.suptitle("Elastic Deform", fontsize=16)
        f.show()
        f.savefig(os.path.join(saved_figure_dir, "elastic_deform_param_cmp.jpg"))

        conclusion_list.append("The combination of Sigma = 10 and points = 3 is a sensible configuration")

    if show_gaussian_nose:
        sigma_list = [1 / 255, 3 / 255, 5 / 255]
        sigma_text_list = [None, "1/255", "3/255", "5/255"]
        f, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=200)

        axes[0][0].imshow(image, **imshow_kwargs)
        axes[0][0].set_title("(a) original image", **axis_title_kwargs)
        set_axis_invisible(axes[0][0])

        for index, sigma in enumerate(sigma_list):
            index += 1
            row = index // 2
            column = index % 2
            image_gn, _, _ = gaussian_noise_np(image, mask, weight_map, sigma)
            axes[row][column].imshow(image_gn, **imshow_kwargs)
            axes[row][column].set_title("({}) sigma = {}".format(letter_list[index],
                                                                 sigma_text_list[index]),
                                        **axis_title_kwargs)
            set_axis_invisible(axes[row][column])
        f.suptitle("Gaussian Noise", fontsize=16)
        f.savefig(os.path.join(saved_figure_dir, "gaussian_noise_param_cmp.jpg"))
        f.show()

        conclusion_list.append("For Gaussian_noise, sigma = 1/255 "
                               "is a sensible configuration")
