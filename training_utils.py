__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"

"""
Bunch of stuff that should make the training do what it is supposed to do
"""


import os
import pandas
import numpy as np
import sys
import time

from keras.callbacks import Callback
from keras import backend as K
from sklearn.metrics import f1_score

class TimingCallback(Callback):
    """
    From https://github.com/keras-team/keras/issues/5105
    """
    def __init__(self):
        self.logs = []
        self.starttime = time.time()
        self.lasttime = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.lasttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        currtime = time.time()
        from_begin = (currtime-self.starttime)/60
        last_epoch = (currtime-self.lasttime)/60
        print("Last epoch has lasted: " + str(last_epoch) + " minutes")
        print("Training is lasting: " + str(from_begin) + " minutes")


class RealValidationCallback(Callback):
    def __init__(self, patience, log_path, file_path):
        self.logs = []
        self.best_score = 0
        self.waited = 0
        self.patience = patience
        self.log = open(log_path, 'w')
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs={}):
        validation_x = self.model.validation_data[0]
        Y_test = self.model.validation_data[1]
        Y_pred = self.model.predict[validation_x]

        Y_pred_prop = np.concatenate([Y_pred[2], Y_pred[3]])
        Y_test_prop = np.concatenate([Y_test[2], Y_test[3]])
        Y_pred_prop = np.argmax(Y_pred_prop, axis=-1)
        Y_test_prop = np.argmax(Y_test_prop, axis=-1)
        Y_pred_links = np.argmax(Y_pred[0], axis=-1)
        Y_test_links = np.argmax(Y_test[0], axis=-1)
        Y_pred_rel = np.argmax(Y_pred[1], axis=-1)
        Y_test_rel = np.argmax(Y_test[1], axis=-1)

        score_link = f1_score(Y_test_links, Y_pred_links, average=None, labels=[0])
        score_rel = f1_score(Y_test_rel, Y_pred_rel, average=None, labels=[0, 2])
        score_rel_AVG = f1_score(Y_test_rel, Y_pred_rel, average='macro', labels=[0, 2])
        score_prop = f1_score(Y_test_prop, Y_pred_prop, average=None)
        score_prop_AVG = f1_score(Y_test_prop, Y_pred_prop, average='macro')

        score_AVG_LP = np.mean([score_link, score_prop_AVG])
        score_AVG_all = np.mean([score_link, score_prop_AVG, score_rel_AVG])

        string = str(epoch) + "\t" + str(score_AVG_all[0]) + "\t" + str(score_AVG_LP[0])
        string += "\t" + str(score_link[0]) + "\t" + str(score_rel_AVG)
        for score in score_rel:
            string += "\t" + str(score)
        string += "\t" + str(score_prop_AVG)
        for score in score_prop:
            string += "\t" + str(score)

        if score_link > self.best_score:
            self.best_score = score_link
            string += "\t!"
            # save
            self.model.save_weights(self.file_path % epoch)
        else:
            self.waited += 1
            # early stopping
            if self.waited > self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        self.log.write(string + "\n")

        self.log.flush()

    def on_train_end(self, logs=None):
        self.log.close()


def create_lr_annealing_function(initial_lr=0.001, k=0.001, fixed_epoch=-1):

    def lr_annealing(epoch, lr=0):
        """
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        if fixed_epoch <= 0:
            lr = (initial_lr / (1 + k * epoch))
        else:
            lr = (initial_lr / (1 + k * fixed_epoch))
        print("\tNEW LR: " + str(lr))

        return lr

    return lr_annealing


"""
def wrong_lr_annealing_function(epoch, initial_lr=0.001, k=0.001, fixed_epoch=-1):

    if fixed_epoch <= 0:
        lr = (initial_lr / (1 + k * epoch))
    else:
        lr = (initial_lr / (1 + k * fixed_epoch))
    print("\tNEW LR: " + str(lr))

    return lr
"""

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


def get_avgF1(indexes):
    """
    Create the average f1-measure for the classes indicated by indexes
    :param indexes: iterable object of ints that represent classes
    :return: the average f1-measure
    """

    def some_class_precision(index, y_true, y_pred):
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

    def some_class_recall(index, y_true, y_pred):
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

    def avgF1(y_true, y_pred):

        fmeasures = []

        for index in indexes:
            index = int(index)
            p = some_class_precision(index, y_true, y_pred)
            r = some_class_recall(index, y_true, y_pred)

            beta = 1
            bb = beta ** 2
            fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())

            fmeasures.append(fbeta_score)

        return K.mean(K.stack(fmeasures))

    if len(indexes)<1:
        raise Exception('NEED MORE INDEXES!')
        return None
    else:
        name = "F1"
        for index in indexes:
            name += "_" + str(index)
        avgF1.__name__ = name
        return avgF1

