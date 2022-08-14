import numpy as np

from model import get_compiled_unet, get_compiled_lightweight_unet, get_compiled_binary_lightweight_unet
from training_utils import evaluate_on_test_set, train_val_test_split


def _get_test_set(target_size=(512, 512), batch_size=1, use_weight_map=False, seed=1):
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
        seed=seed, num_folds=5, fold_index=0)
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
    test_set = _get_test_set()
    metric_dict = evaluate_on_test_set(baseline_unet, test_set)
    return metric_dict


def script_evaluate_fine_tuned_vanilla_unet(fold_index):
    if fold_index == 0:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_0/" \
                              "vanilla_unet-fine_tuned-fold_0.h5"
    elif fold_index == 1:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_1/" \
                              "vanilla_unet-fine_tuned-fold_1.h5"
    elif fold_index == 2:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_2/" \
                              "vanilla_unet-fine_tuned-fold_2.h5"
    elif fold_index == 3:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_3/" \
                              "vanilla_unet-fine_tuned-fold_3.h5"

    elif fold_index == 4:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-10_fine_tune_vanilla_unet_with_l2-fold_4/" \
                              "vanilla_unet-fine_tuned-fold_4.h5"
    else:
        raise RuntimeError("unavailable fold index")

    vanilla_unet = get_compiled_unet((512, 512, 1), pretrained_weights=trained_weight_path)
    test_set = _get_test_set()

    metric_dict = evaluate_on_test_set(vanilla_unet, test_set)
    return metric_dict


def script_evaluate_lightweight_unet(fold_index):
    if fold_index == 0:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-11_retrain_lw_unet_after_knowledge_distillation-fold_0/" \
                              "lw_unet_retrained_after_knowledge_distillation-fold_0.h5"
    elif fold_index == 1:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-11_retrain_lw_unet_after_knowledge_distillation-fold_1/" \
                              "lw_unet_retrained_after_knowledge_distillation-fold_1.h5"
    elif fold_index == 2:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-11_retrain_lw_unet_after_knowledge_distillation-fold_2/" \
                              "lw_unet_retrained_after_knowledge_distillation-fold_2.h5"
    elif fold_index == 3:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-11_retrain_lw_unet_after_knowledge_distillation-fold_3/" \
                              "lw_unet_retrained_after_knowledge_distillation-fold_3.h5"
    elif fold_index == 4:
        trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                              "2022-08-11_retrain_lw_unet_after_knowledge_distillation-fold_4/" \
                              "lw_unet_retrained_after_knowledge_distillation-fold_4.h5"
    else:
        raise RuntimeError("unavailable fold index")

    lw_unet = get_compiled_lightweight_unet((512, 512, 1), pretrained_weight=trained_weight_path)
    test_set = _get_test_set()

    metric_dict = evaluate_on_test_set(lw_unet, test_set)
    return metric_dict


def script_evaluate_blu_net(fold_index):
    match fold_index:
        case 0:
            trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-12_blu_net-residual_binarize_retrained_lw_unet-a3_d3_p3_c3-fold_0/" \
                                  "blu_unet-a3_d3_p3_c3-fold_0.h5"
        case 1:
            trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-12_blu_net-residual_binarize_retrained_lw_unet-a3_d3_p3_c3-fold_1/" \
                                  "blu_unet-a3_d3_p3_c3-fold_1.h5"
        case 2:
            trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-12_blu_net-residual_binarize_retrained_lw_unet-a3_d3_p3_c3-fold_2/" \
                                  "blu_unet-a3_d3_p3_c3-fold_2.h5"
        case 3:
            trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-12_blu_net-residual_binarize_retrained_lw_unet-a3_d3_p3_c3-fold_3/" \
                                  "blu_unet-a3_d3_p3_c3-fold_3.h5"
        case 4:
            trained_weight_path = "E:/ED_MS/Semester_3/Codes/MyProject/tensorboard_logs/" \
                                  "2022-08-12_blu_net-residual_binarize_retrained_lw_unet-a3_d3_p3_c3-fold_4/" \
                                  "blu_unet-a3_d3_p3_c3-fold_4.h5"
        case _:
            raise RuntimeError("unavailable fold _index")

    blu_net = get_compiled_binary_lightweight_unet((512, 512, 1), pretrained_weight=trained_weight_path)
    test_set = _get_test_set()

    metric_dict = evaluate_on_test_set(blu_net, test_set)

    return metric_dict


if __name__ == '__main__':
    # * - [x] evaluate baseline_unet on 5 folds
    # * - [x] evaluate vanilla_unet on 5 folds
    # * - [x] evaluate lightweight_unet on 5 folds
    # * - [x] evaluate blu_net oon 5 folds

    # *: baseline unet
    # baseline_unet, binary_iou = 0.04680933430790901, binary_f1score = 0.08308295905590057
    # average_5_folds(name="baseline_unet", one_fold_script=script_evaluate_baseline_vanilla_unet)

    # *: vanilla unet
    # fine_tuned_vanilla_unet, binary_iou = 0.8869273066520691, binary_f1score = 0.939757227897644
    # average_5_folds(name="fine_tuned_vanilla_unet", one_fold_script=script_evaluate_fine_tuned_vanilla_unet)

    # *: lightweight unet
    # lw_unet_knowledge_distillation, binary_iou = 0.8941400647163391, binary_f1score = 0.9438070058822632
    # average_5_folds(name="lw_unet_knowledge_distillation", one_fold_script=script_evaluate_lightweight_unet)

    # *: blu_net
    # blu_net, binary_iou = 0.8531662821769714, binary_f1score = 0.9202474355697632
    average_5_folds("blu_net", script_evaluate_blu_net)
