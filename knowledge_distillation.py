import time

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import metrics

from tqdm import tqdm

from training_utils import binarize_and_compute_iou


class KnowledgeDistillation:
    def __init__(self, name, teacher, student, *args, **kwargs):
        self.teacher = teacher
        self.student = student
        self.student_optimizer = None
        self.student_loss_fn = None
        self.current_step_loss = None
        self.current_step_iou = None

    def compile(self, optimizer, loss_fn):
        self.student_optimizer = optimizer
        self.student_loss_fn = loss_fn

    def call(self, inputs, training=None, mask=None):
        student_outputs = self.teacher(inputs=inputs, training=training)
        teacher_outputs = self.student(inputs=inputs, training=training)
        return student_outputs, teacher_outputs

    def train_step(self, image_batch, mask_batch):
        teacher_outputs = self.teacher(image_batch, training=False)
        with tf.GradientTape() as tape:
            student_outputs = self.student(image_batch, training=True)
            student_loss = 0.0
            for student_output, teacher_output in zip(student_outputs, teacher_outputs):
                student_loss += self.student_loss_fn(teacher_output, student_output)

        grads = tape.gradient(student_loss, self.student.trainable_variables)
        self.student_optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.current_step_loss = student_loss.numpy()
        self.current_step_iou = binarize_and_compute_iou(mask_batch, student_outputs[-1]).numpy()

        return self.current_step_loss, self.current_step_iou


def distill_knowledge(knowledge_distillation: KnowledgeDistillation, start_epoch, end_epoch, train_set, val_set,
                      checkpoint_path, logdir):
    train_dict = {"epoch": [], "loss": [], "binary_IoU": [], "val_binary_IoU": []}
    train_step_count = 0
    for epoch in range(start_epoch, end_epoch):
        # train
        train_summary_writer = tf.summary.create_file_writer(logdir + "/train")
        with train_summary_writer.as_default():
            loss_list = []
            binary_iou_list = []
            for step, (image_batch, mask_batch) in tqdm(enumerate(train_set), total=len(train_set)):
                loss, binary_iou = knowledge_distillation.train_step(image_batch, mask_batch)
                loss_list.append(loss)
                binary_iou_list.append(binary_iou)

                train_step_count += 1

            epoch_loss = np.mean(loss_list)
            epoch_binary_iou = np.mean(binary_iou_list)
            tf.summary.scalar("epoch_loss", epoch_loss, epoch)
            tf.summary.scalar("epoch_binary_IoU", epoch_binary_iou, epoch)
            # tf.summary.scalar("learning_rate", knowledge_distillation.student_optimizer.lr.current_lr, epoch)
            train_dict["epoch"].append(epoch)
            train_dict["loss"].append(epoch_loss)
            train_dict["binary_IoU"].append(epoch_binary_iou)

        # val
        val_summary_writer = tf.summary.create_file_writer(logdir + "/validation")
        with val_summary_writer.as_default():
            val_iou_list = []
            for image_batch, mask_batch in val_set:
                pred_mask = knowledge_distillation.student(image_batch, training=False)[-1]
                val_iou_list.append(binarize_and_compute_iou(y_true=mask_batch, y_pred=pred_mask).numpy())

            epoch_val_binary_iou = np.mean(val_iou_list)
            tf.summary.scalar("epoch_binary_IoU", epoch_val_binary_iou, epoch)
            train_dict["val_binary_IoU"].append(epoch_val_binary_iou)

        print("epoch: {}, loss = {}, binary_IoU = {}, "
              "val_binary_IoU = {}, lr = {}".format(epoch,
                                                    epoch_loss,
                                                    epoch_binary_iou,
                                                    epoch_val_binary_iou,
                                                    knowledge_distillation.student_optimizer.lr.current_lr))

        # take a break
        if epoch != 0 and epoch % 40 == 0:
            time.sleep(120)

    knowledge_distillation.student.save_weights(filepath=checkpoint_path, save_format="h5")
    df = pd.DataFrame(train_dict)
    return df
