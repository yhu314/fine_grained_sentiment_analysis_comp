import keras.backend as K
import tensorflow as tf


def f1(y_true, y_pred):
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 2 * precision * recall / (1 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)


def f1_macro(y_true, y_pred):
    n_classes = y_true.shape[1]
    sum_f1 = 0
    for idx in range(n_classes):
        y_true_class = y_true[:, idx]
        y_pred_class = y_pred[:, idx]
        sum_f1 = f1(y_true_class, y_pred_class)
    return sum_f1/n_classes
