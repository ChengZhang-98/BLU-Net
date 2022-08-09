import numpy as np

from training_utils import evaluate_on_test_set, train_val_test_split
from model import get_compiled_unet, get_compiled_lightweight_unet, get_compiled_binary_lightweight_unet


def _get_test_set(fold_index, target_size=(512, 512), batch_size=1, use_weight_map=False, seed=1):
    image_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Annotated"
    image_type = "tif"
    mask_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Masks"
    mask_type = "tif"
    weight_map_dir = "E:/ED_MS/Semester_3/Dataset/DIC_Set/DIC_Set1_Weights"
    weight_map_type = "npy"
    dataset_name = "DIC"

    _, _, test_set = train_val_test_split(
        batch_size=batch_size, dataset_name="DIC", use_weight_map=use_weight_map,
        image_dir=image_dir, image_type=image_type, mask_dir=mask_dir, mask_type=mask_type,
        weight_map_dir=weight_map_dir, weight_map_type=weight_map_type,
        target_size=target_size, data_aug_transform=None,
        seed=seed, num_folds=5, fold_index=fold_index)
    return test_set


def average_5_folds(name, one_fold_script):
    binary_iou_list = []
    binary_f1score_list = []
    for fold_index in range(5):
        fold_metric_dict = one_fold_script(fold_index)
        binary_iou_list.append(fold_metric_dict["binary_iou"])
        binary_f1score_list.append(fold_metric_dict["binary_f1score"])

    print("{}, binary_iou = {}, "
          "binary_f1score = {}".format(name, np.mean(binary_iou_list), np.mean(binary_f1score_list)))


def script_evaluate_baseline_vanilla_unet(fold_index):
    weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/checkpoints/trained_weights/unet_agarpads_seg_evaluation2.hdf5"
    baseline_unet = get_compiled_unet((512, 512, 1), pretrained_weights=weight_path)
    test_set = _get_test_set(fold_index)
    metric_dict = evaluate_on_test_set(baseline_unet, test_set)
    return metric_dict


def script_evaluate_fine_tuned_vanilla_unet(fold_index):
    trained_weight_path = None
    if fold_index == 0:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-06_fine_tune_vanilla_unet_with_l2-fold_0/" \
                              "vanilla_unet-fine-tuned_IoU=0.8880.h5"
    elif fold_index == 1:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-07_fine_tune_vanilla_unet_with_l2-fold_1/" \
                              "vanilla_unet-fine_tuned-fold_1-end_epoch-IoU=0.9125.h5"
    elif fold_index == 2:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-07_fine_tune_vanilla_unet_with_l2-fold_2/" \
                              "vanilla_unet-fine_tuned-fold_2-IoU=0.901637.h5"
    elif fold_index == 3:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-07_fine_tune_vanilla_unet_with_l2-fold_3/" \
                              "vanilla_unet-fine_tuned-fold_3-IoU=0.927555.h5"
    elif fold_index == 4:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-07_fine_tune_vanilla_unet_with_l2-fold_4/" \
                              "vanilla_unet-fine_tuned-fold_4-IoU=0.885986.h5"
    else:
        raise RuntimeError("unavailable fold index")

    vanilla_unet = get_compiled_unet((512, 512, 1), pretrained_weights=trained_weight_path)
    test_set = _get_test_set(fold_index)

    metric_dict = evaluate_on_test_set(vanilla_unet, test_set)
    return metric_dict


if __name__ == '__main__':
    # todo list
    # ! - [x] evaluate baseline_unet on 5 folds
    # ! - [x] evaluate vanilla_unet on 5 folds
    # ! - [ ] evaluate lightweight_unet on 5 folds
    # ! - [ ] evaluate blu_net oon 5 folds

    # *: baseline unet
    # binary_iou = 0.8919236063957214, binary_f1score = 0.9426580667495728
    average_5_folds(name="baseline_unet", one_fold_script=script_evaluate_baseline_vanilla_unet)

    # *: vanilla unet
    average_5_folds(name="fine_tuned_vanilla_unet", one_fold_script=script_evaluate_fine_tuned_vanilla_unet)
