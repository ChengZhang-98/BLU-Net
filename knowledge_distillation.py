import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics


# todo: untested
from tqdm import tqdm

from training_utils import CustomBinaryIoU


class KnowledgeDistillation:
    def __init__(self, name, teacher, student):
        self.teacher = teacher
        self.student = student
        self.student_optimizer = None
        self.student_code_loss_fn = None
        self.student_pred_loss_fn = None
        self.student_code_metric_tracker = metrics.Mean(name="code_l1_error")
        self.student_pred_metric_tracker = metrics.Mean(name="pred_l1_error")
        self.lambda_code = None

    def compile(self, optimizer, code_loss_fn, pred_loss_fn, lambda_code=1):
        self.student_optimizer = optimizer
        self.student_code_loss_fn = code_loss_fn
        self.student_pred_loss_fn = pred_loss_fn
        self.lambda_code = lambda_code

    def call(self, inputs, training=None, mask=None):
        teacher_code, teacher_pred = self.teacher(inputs=inputs, training=training)
        student_code, student_pred = self.student(inputs=inputs, training=training)
        return student_code, student_pred, teacher_code, teacher_pred

    def train_step(self, image_batch, mask_batch):
        teacher_code, teacher_pred = self.teacher(image_batch, training=False)
        with tf.GradientTape() as tape:
            student_code, student_pred = self.student(image_batch, training=True)
            code_loss = self.student_code_loss_fn(teacher_code, student_code)
            pred_loss = self.student_pred_loss_fn(teacher_pred, student_pred)
            loss = pred_loss + self.lambda_code * code_loss
        grads = tape.gradient(loss, self.student.trainable_variables)
        self.student_optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.student_code_metric_tracker.update_state(code_loss)
        self.student_pred_metric_tracker.update_state(pred_loss)

        return dict(code_loss=self.student_code_metric_tracker.result(),
                    pred_loss=self.student_pred_metric_tracker.result())

    @property
    def metrics(self):
        return [self.student_code_metric_tracker, self.student_pred_metric_tracker]


def distill_knowledge(knowledge_distillation, start_epoch, end_epoch, train_set, val_set, checkpoint_path):
    train_dict = {"epoch": [], "code_loss": [], "pred_loss": [], "val_binary_IoU": []}
    for epoch in range(start_epoch, end_epoch):
        train_code_loss_list = []
        train_pred_loss_list = []
        for step, (image_batch, mask_batch) in tqdm(enumerate(train_set), total=len(train_set)):
            loss_dict = knowledge_distillation.train_step(image_batch, mask_batch)
            train_code_loss_list.append(loss_dict["code_loss"])
            train_pred_loss_list.append(loss_dict["pred_loss"])

        epoch_code_loss = np.mean(train_code_loss_list)
        epoch_pred_loss = np.mean(train_pred_loss_list)
        train_dict["epoch"].append(epoch)
        train_dict["code_loss"].append(epoch_code_loss)
        train_dict["pred_loss"].append(epoch_pred_loss)

        val_iou_list = []
        metric_iou = CustomBinaryIoU()

        for image_batch, mask_batch in val_set:
            _, pred_mask = knowledge_distillation.student(image_batch, training=False)
            metric_iou.update_state(mask_batch, pred_mask)
            val_iou_list.append(metric_iou.result())
        epoch_val_binary_iou = np.mean(val_iou_list)
        train_dict["val_binary_IoU"].append(epoch_val_binary_iou)

        print("epoch: {}, code_loss = {}, pred_loss = {}, val_binary_IoU = {}".format(epoch, epoch_code_loss,
                                                                                      epoch_pred_loss,
                                                                                      epoch_val_binary_iou))

        # take a break
        if epoch != 0 and epoch % 40 == 0:
            time.sleep(120)

    knowledge_distillation.student.save_weights(filepath=checkpoint_path, save_format="h5")
    df = pd.DataFrame(train_dict)
    return df
