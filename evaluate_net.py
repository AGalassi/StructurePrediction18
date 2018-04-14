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

from networks import build_net_1

from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing.sequence import  pad_sequences
from training_utils import TimingCallback, lr_annealing, fmeasure, get_single_class_fmeasure, get_avgF1
from training import load_dataset
from glove_loader import DIM

if __name__ == '__main__':

    name = 'prova'
    if len(sys.argv) > 1:
        name = sys.argv[1]
    print(str(time.ctime()) + "\tLAUNCHING TRAINING: " + name)

    epochs = 1000

    dataset_name = 'cdcp_ACL17'
    dataset_version = 'new_2'


    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = load_dataset(dataset_name=dataset_name,
                                                       dataset_version=dataset_version,
                                                       dataset_split='total')
    print(str(time.ctime()) + "\tDATASET LOADED...")

    sys.stdout.flush()

    print(str(time.ctime()) + "\tPROCESSING DATA AND MODEL...")

    split = 'test'
    X_text_test = dataset[split]['texts']
    dataset[split]['texts'] = 0
    X_source_test = dataset[split]['source_props']
    dataset[split]['source_props'] = 0
    X_target_test = dataset[split]['target_props']
    dataset[split]['target_props'] = 0
    Y_links_test = np.array(dataset[split]['links'])
    Y_rtype_test = np.array(dataset[split]['relations_type'])
    Y_stype_test = np.array(dataset[split]['sources_type'])
    Y_ttype_test = np.array(dataset[split]['targets_type'])
    X_test = [X_text_test, X_source_test, X_target_test]
    Y_test = [Y_links_test, Y_rtype_test, Y_stype_test, Y_ttype_test]

    print(str(time.ctime()) + "\t\tTEST DATA PROCESSED...")

    fmeasure_1 = get_single_class_fmeasure(0)
    fmeasure_2 = get_avgF1([0, 1, 2, 3])

    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)

    last_path = ""
    for epoch in range(epochs, 1, -1):
        netpath = os.path.join(save_dir, name + '_completemodel.%03d.h5' % epoch)
        if os.path.exists(netpath):
            last_path = netpath
            break

    print("PATH:\t" + last_path)

    model = load_model(last_path,
                       custom_objects={'fmeasure': fmeasure,
                                       'single_class_fmeasure': fmeasure_1,
                                       'fmeasure_some_classes': fmeasure_2,
                                       }
                       )

    score = model.evaluate(X_test, Y_test, verbose=0)

    string = ""
    for metric in model.metrics_names:
        string += metric + "\t"
    print(string)

    string = ""
    for metric in score:
        string += str(metric) + "\t"
    print(string)

    sys.stdout.flush()

