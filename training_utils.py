__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"

import os
import pandas
import numpy as np
import sys
import time

from keras.callbacks import Callback
from keras import backend as K

class TimingCallback(Callback):
    """
    From https://github.com/keras-team/keras/issues/5105
    """
    def __init__(self):
        self.logs = []
        self.starttime = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.lasttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        currtime = time.time()
        from_begin = (currtime-self.starttime)/60
        last_epoch = (currtime-self.lasttime)/60
        print("Last epoch has lasted: " + str(last_epoch))
        print("Training is lasting: " + str(from_begin))



def lr_annealing(epoch, initial_lr=0.001, k=0.001):
    """
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """

    lr = (initial_lr / (1 + k * epoch))

    return lr


def precision(y_true, y_pred):
    """
    From: https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7#diff-7b49e1c42728a58a9d08643a79f44cd4
    Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """
    From: https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7#diff-7b49e1c42728a58a9d08643a79f44cd4
    Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """
    From: https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7#diff-7b49e1c42728a58a9d08643a79f44cd4
    Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    """
    From: https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7#diff-7b49e1c42728a58a9d08643a79f44cd4
    Computes the f-measure, the harmonic mean of precision and recall.

    Here it is only computed as a batch-wise average, not globally.
    """

    return fbeta_score(y_true, y_pred, beta=1)


def get_single_class_fmeasure(index):
    """
    Create a fmeasure only for the class of value index
    :param index:
    :return:
    """

    def single_class_precision(y_true, y_pred):
        """
        Based on https://stackoverflow.com/a/41717938/5464787
        :param y_true:
        :param y_pred:
        :return:
        """
        # true classes
        class_id_true = K.argmax(y_true, axis=-1)
        # predicted classes
        class_id_preds = K.argmax(y_pred, axis=-1)
        # predictions of the interested class (true positives + false positives)
        mask = K.cast(K.equal(class_id_preds, index), 'int32')
        # right predictions (true positives + true negatives)
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32')
        # true positives
        masked_tensor = class_acc_tensor * mask
        class_acc = K.sum(masked_tensor) / K.maximum(K.sum(mask), 1)
        return class_acc

    def single_class_recall(y_true, y_pred):
        """
        Based on https://stackoverflow.com/a/41717938/5464787
        :param y_true:
        :param y_pred:
        :return:
        """
        # true classes
        class_id_true = K.argmax(y_true, axis=-1)
        # predicted classes
        class_id_preds = K.argmax(y_pred, axis=-1)
        # true of interested class (true positives + false negatives)
        mask = K.cast(K.equal(class_id_true, index), 'int32')
        # right predictions (true positives + true negatives)
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32')
        # true positives
        masked_tensor = class_acc_tensor * mask
        class_rec = K.sum(masked_tensor) / K.maximum(K.sum(mask), 1)
        return class_rec

    def single_class_fmeasure(y_true, y_pred):
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = single_class_precision(y_true, y_pred)
        r = single_class_recall(y_true, y_pred)

        beta = 1
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

        return fbeta_score

    return single_class_fmeasure


def get_fmeasure_some_classes(indexes):
    """
    Create a fmeasure only for the class of value index
    :param index:
    :return:
    """


    def some_class_precision(y_true, y_pred):
        """
        Based on https://stackoverflow.com/a/41717938/5464787
        :param y_true:
        :param y_pred:
        :return:
        """
        # true classes
        class_id_true = K.argmax(y_true, axis=-1)
        # predicted classes
        class_id_preds = K.argmax(y_pred, axis=-1)

        # predictions of the interested class (true positives + false positives)
        mask = K.cast(K.equal(class_id_preds, indexes[0]), 'int32')
        for index in indexes[1:]:
            m = K.cast(K.equal(class_id_preds, index), 'int32')
            mask = m + mask

        # right predictions (true positives + true negatives)
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32')
        # true positives
        masked_tensor = class_acc_tensor * mask
        class_acc = K.sum(masked_tensor) / K.maximum(K.sum(mask), 1)
        return class_acc

    def some_class_recall(y_true, y_pred):
        """
        Based on https://stackoverflow.com/a/41717938/5464787
        :param y_true:
        :param y_pred:
        :return:
        """
        # true classes
        class_id_true = K.argmax(y_true, axis=-1)
        # predicted classes
        class_id_preds = K.argmax(y_pred, axis=-1)

        # true of interested class (true positives + false negatives)
        mask = K.cast(K.equal(class_id_true, indexes[0]), 'int32')
        for index in indexes[1:]:
            m = K.cast(K.equal(class_id_true, index), 'int32')
            mask = m + mask

        # right predictions (true positives + true negatives)
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32')
        # true positives
        masked_tensor = class_acc_tensor * mask
        class_rec = K.sum(masked_tensor) / K.maximum(K.sum(mask), 1)
        return class_rec

    def fmeasure_some_classes(y_true, y_pred):
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = some_class_precision(y_true, y_pred)
        r = some_class_recall(y_true, y_pred)

        beta = 1
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

        return fbeta_score


    if len(indexes)<1:
        raise Exception('NEED MORE INDEXES!')
        return None
    else:
        return fmeasure_some_classes

