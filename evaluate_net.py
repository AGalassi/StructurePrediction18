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
from training_utils import TimingCallback, lr_annealing, fmeasure, get_avgF1
from training import load_dataset
from sklearn.metrics import f1_score
from glove_loader import DIM


if __name__ == '__main__':

    name = 'prova'
    if len(sys.argv) > 1:
        name = sys.argv[1]
    feature_type = 'bow'

    epochs = 1000

    name = 'cdcpN6'
    dataset_name = 'cdcp_ACL17'
    dataset_version = 'new_3'
    split = 'test'
    epoch = None
    # epoch = 173

    # determine which network to load
    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)
    last_path = ""
    if epoch is None:
        for epoch in range(epochs, 0, -1):
            netpath = os.path.join(save_dir, name + '_completemodel.%03d.h5' % epoch)
            if os.path.exists(netpath):
                last_path = netpath
                break
    else:
        last_path = os.path.join(save_dir, name + '_completemodel.%03d.h5' % epoch)

    print(str(time.ctime()) + "\tTESTING NET: " + name)
    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = load_dataset(dataset_name=dataset_name,
                                                       dataset_version=dataset_version,
                                                       dataset_split=split,
                                                       feature_type=feature_type,
                                                       min_prop_len=153,
                                                       min_text_len=552)
    print(str(time.ctime()) + "\tDATASET LOADED...")

    sys.stdout.flush()

    print(str(time.ctime()) + "\tPROCESSING DATA AND MODEL...")
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

    X_marks_test = dataset[split]['mark']
    X_dist_test = dataset[split]['distance']
    X3_test = [X_text_test, X_source_test, X_target_test, X_dist_test, X_marks_test]

    print(str(time.ctime()) + "\t\tTEST DATA PROCESSED...")

    print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

    # create metrics for the model
    fmeasure_0 = get_avgF1([0])
    fmeasure_1 = get_avgF1([1])
    fmeasure_2 = get_avgF1([2])
    fmeasure_3 = get_avgF1([3])
    fmeasure_4 = get_avgF1([4])
    fmeasure_0_1_2_3 = get_avgF1([0, 1, 2, 3])
    fmeasure_0_1_2_3_4 = get_avgF1([0, 1, 2, 3, 4])
    fmeasure_0_2 = get_avgF1([0, 2])
    fmeasure_0_1_2 = get_avgF1([0, 1, 2])
    fmeasure_0_1 = get_avgF1([0, 1, 2])

    fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_3, fmeasure_4, fmeasure_0_1_2_3, fmeasure_0_1_2_3_4,
                 fmeasure_0_2, fmeasure_0_1_2, fmeasure_0_1]

    props_fmeasures = []

    if dataset_name == 'cdcp_ACL17':
        props_fmeasures = [fmeasure_0_1_2_3_4]
    # elif dataset_name == 'AAEC_v2':
    #    props_fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_0_1_2]
    elif dataset_name == 'AAEC_v2':
        props_fmeasures = [fmeasure_0_1_2]

    # for using them during model loading
    fmeasures_names = {}
    for fmeasure in fmeasures:
        fmeasures_names[fmeasure.__name__] = fmeasure

    model = load_model(last_path,
                       custom_objects=fmeasures_names
                       )


    print(str(time.ctime()) + "\t\tNETWORK LOADED")

    testfile = open(os.path.join(save_dir, name + ".%03d_eval.txt" % epoch), 'w')


    print("\n-----------------------\n")

    print(str(time.ctime()) + "\tEVALUATING MODEL")

    # evaluate the model with sci-kit learn

    if dataset_name == "AAEC_v2":
        testfile.write("\n\nset\tAVG all\tAVG LP\tlink\tR AVG dir\tR support\tR attack\t" +
                       "P AVG\tP premise\tP claim\tP major claim\n")
    elif dataset_name == "cdcp_ACL17":
        testfile.write("\n\nset\tAVG all\tAVG LP\tlink\tR AVG dir\tR reason\tR evidence\t" +
                       "P AVG\tP policy\tP fact\tP testimony\tP value\tP reference\n")

    Y_pred = model.predict(X3_test)

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

    string = split + "\t" + str(score_AVG_all[0]) + "\t" + str(score_AVG_LP[0])
    string += "\t" + str(score_link[0]) + "\t" + str(score_rel_AVG)
    for score in score_rel:
        string += "\t" + str(score)
    string += "\t" + str(score_prop_AVG)
    for score in score_prop:
        string += "\t" + str(score)

    testfile.write(string + "\n")

    testfile.flush()
    testfile.close()

    print(str(time.ctime()) + "\t\tFINISHED")

