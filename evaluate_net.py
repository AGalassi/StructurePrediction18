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
import json
import training

from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model, model_from_json
from keras.preprocessing.sequence import  pad_sequences
from training_utils import TimingCallback, fmeasure, get_avgF1
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from glove_loader import DIM

import networks


MAXEPOCHS = 1000
MAXITERATIONS = 20


def evaluate_single_epoch(netname, dataset_name, dataset_version, split='test', epoch=None, feature_type='bow'):

    epochs = MAXEPOCHS

    # determine which network to load
    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, netname)
    last_path = ""
    if epoch is None:
        for epoch in range(epochs, 0, -1):
            netpath = os.path.join(save_dir, netname + '_completemodel.%03d.h5' % epoch)
            if os.path.exists(netpath):
                last_path = netpath
                break
    else:
        last_path = os.path.join(save_dir, netname + '_completemodel.%03d.h5' % epoch)

    print(str(time.ctime()) + "\tTESTING NET: " + netname)
    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = training.load_dataset(dataset_name=dataset_name,
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

    testfile = open(os.path.join(save_dir, netname + ".%03d_eval.txt" % epoch), 'a')


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


# TODO: optimize (test evaluation outside of cycle
def evaluate_every_epoch(netname, dataset_name, dataset_version, split='test', feature_type='bow',
                         context=True, distance=True):
    """
    Evaluate the network in each of the available epochs
    :param netname:
    :param dataset_name:
    :param dataset_version:
    :param split:
    :param feature_type:
    :return:
    """


    if dataset_name == 'AAEC_v2':
        min_text = 168
        min_prop = 72
    else:
        min_text = 552
        min_prop = 153

    epochs = MAXEPOCHS

    print(str(time.ctime()) + "\tTESTING NET: " + netname)
    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = training.load_dataset(dataset_name=dataset_name,
                                                       dataset_version=dataset_version,
                                                       dataset_split=split,
                                                       feature_type=feature_type,
                                                       min_prop_len=min_prop,
                                                       min_text_len=min_text)
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
    
    if context and distance:
        X3_test = [X_text_test, X_source_test, X_target_test, X_dist_test, X_marks_test]
    elif distance:
        X3_test = [X_source_test, X_target_test, X_dist_test]
    elif context:
        X3_test = [X_text_test, X_source_test, X_target_test, X_marks_test]
    else:
        X3_test = [X_source_test, X_target_test]

    print(str(time.ctime()) + "\t\tTEST DATA PROCESSED...")

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

    crop0 = networks.create_crop_fn(1, 0, 1)
    crop1 = networks.create_crop_fn(1, 1, 2)
    crop2 = networks.create_crop_fn(1, 2, 3)
    crop3 = networks.create_crop_fn(1, 3, 4)
    crop4 = networks.create_crop_fn(1, 4, 5)

    fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_3, fmeasure_4, fmeasure_0_1_2_3, fmeasure_0_1_2_3_4,
                 fmeasure_0_2, fmeasure_0_1_2, fmeasure_0_1]

    crops = [crop0, crop1, crop2, crop3, crop4]


    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure
    for crop in crops:
        custom_objects[crop.__name__] = crop


    # determine which network to load
    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, netname)
    testfile = open(os.path.join(save_dir, netname + ".history_eval.txt"), 'a')

    if dataset_name == "AAEC_v2":
        testfile.write("epoch\tset\tAVG all\tAVG LP\tlink\tR AVG dir\tR support\tR attack\t" +
                       "P AVG\tP premise\tP claim\tP major claim\n")
    elif dataset_name == "cdcp_ACL17":
        testfile.write("epoch\tset\tAVG all\tAVG LP\tlink\tR AVG dir\tR reason\tR evidence\t" +
                       "P AVG\tP policy\tP fact\tP testimony\tP value\tP reference\n")

    last_path = ""

    model_path = os.path.join(save_dir, netname + '_model.json')
    save_weights_only = False
    if os.path.exists(model_path):
        save_weights_only = True

        with open(model_path, "r") as f:
            string = json.load(f)
            model = model_from_json(string, custom_objects=custom_objects)

    for epoch in range(epochs, 0, -1):
        if save_weights_only:
            netpath = os.path.join(save_dir, netname + '_weights.%03d.h5' % epoch)
        else:
            netpath = os.path.join(save_dir, netname + '_completemodel.%03d.h5' % epoch)
        if os.path.exists(netpath):
            last_path = netpath

            print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

            if save_weights_only:
                model.load_weights(last_path)
            else:
                model = load_model(last_path, custom_objects=custom_objects)

            print(str(time.ctime()) + "\t\tNETWORK LOADED")

            print("\n- - - - - - - - - -\n")

            print(str(time.ctime()) + "\tEVALUATING MODEL")
            sys.stdout.flush()

            # evaluate the model with sci-kit learn

            # 2 dim
            # ax0 = samples
            # ax1 = classes
            Y_pred = model.predict(X3_test)

            # begin of the evaluation of the single propositions scores
            sids = dataset[split]['s_id']
            tids = dataset[split]['t_id']

            # dic
            # each values has 2 dim
            # ax0: ids
            # ax1: classes
            s_pred_scores = {}
            s_test_scores = {}
            t_pred_scores = {}
            t_test_scores = {}

            for index in range(len(sids)):
                sid = sids[index]
                tid = tids[index]

                if sid not in s_pred_scores.keys():
                    s_pred_scores[sid] = []
                    s_test_scores[sid] = []
                s_pred_scores[sid].append(Y_pred[2][index])
                s_test_scores[sid].append(Y[split][2][index])

                if tid not in t_pred_scores.keys():
                    t_pred_scores[tid] = []
                    t_test_scores[tid] = []
                t_pred_scores[tid].append(Y_pred[3][index])
                t_test_scores[tid].append(Y[split][3][index])

            Y_pred_prop_real_list = []
            Y_test_prop_real_list = []

            for p_id in t_pred_scores.keys():
                Y_pred_prop_real_list.append(np.concatenate([s_pred_scores[p_id], t_pred_scores[p_id]]))
                Y_test_prop_real_list.append(np.concatenate([s_test_scores[p_id], t_test_scores[p_id]]))

            # 3 dim
            # ax0: ids
            # ax1: samples
            # ax2: classes
            Y_pred_prop_real_list = np.array(Y_pred_prop_real_list)
            Y_test_prop_real_list = np.array(Y_test_prop_real_list)

            Y_pred_prop_real = []
            Y_test_prop_real = []
            for index in range(len(Y_pred_prop_real_list)):
                Y_pred_prop_real.append(np.sum(Y_pred_prop_real_list[index], axis=-2))
                Y_test_prop_real.append(np.sum(Y_test_prop_real_list[index], axis=-2))

            Y_pred_prop_real = np.array(Y_pred_prop_real)
            Y_test_prop_real = np.array(Y_test_prop_real)
            Y_pred_prop_real = np.argmax(Y_pred_prop_real, axis=-1)
            Y_test_prop_real = np.argmax(Y_test_prop_real, axis=-1)

            # end of the evaluation of the single propositions scores

            # Y_pred_prop = np.concatenate([Y_pred[2], Y_pred[3]])
            # Y_test_prop = np.concatenate([Y[split][2], Y[split][3]])

            # Y_pred_prop = np.argmax(Y_pred_prop, axis=-1)
            # Y_test_prop = np.argmax(Y_test_prop, axis=-1)

            Y_pred_links = np.argmax(Y_pred[0], axis=-1)
            Y_test_links = np.argmax(Y[split][0], axis=-1)

            Y_pred_rel = np.argmax(Y_pred[1], axis=-1)
            Y_test_rel = np.argmax(Y[split][1], axis=-1)

            score_link = f1_score(Y_test_links, Y_pred_links, average=None, labels=[0])
            score_rel = f1_score(Y_test_rel, Y_pred_rel, average=None, labels=[0, 2])
            score_rel_AVG = f1_score(Y_test_rel, Y_pred_rel, average='macro', labels=[0, 2])
            # score_prop = f1_score(Y_test_prop, Y_pred_prop, average=None)
            # score_prop_AVG = f1_score(Y_test_prop, Y_pred_prop, average='macro')

            # score_AVG_LP = np.mean([score_link, score_prop_AVG])
            # score_AVG_all = np.mean([score_link, score_prop_AVG, score_rel_AVG])

            score_prop_real = f1_score(Y_test_prop_real, Y_pred_prop_real, average=None)
            score_prop_AVG_real = f1_score(Y_test_prop_real, Y_pred_prop_real, average='macro')

            score_AVG_LP_real = np.mean([score_link, score_prop_AVG_real])
            score_AVG_all_real = np.mean([score_link, score_prop_AVG_real, score_rel_AVG])

            # string = split + "\t" + str(score_AVG_all[0]) + "\t" + str(score_AVG_LP[0])
            # string += "\t" + str(score_link[0]) + "\t" + str(score_rel_AVG)
            string = split + "_" + str(epoch) + "\t" + str(score_AVG_all_real[0]) + "\t" + str(score_AVG_LP_real[0])
            string += "\t" + str(score_link[0]) + "\t" + str(score_rel_AVG)
            for score in score_rel:
                string += "\t" + str(score)
            string += "\t" + str(score_prop_AVG_real)
            # for score in score_prop:
            #     string += "\t" + str(score)
            for score in score_prop_real:
                string += "\t" + str(score)

            testfile.write(string + "\n")

            testfile.flush()

    testfile.close()

    print(str(time.ctime()) + "\t\tFINISHED")
    print("\n-----------------------\n")



def print_model(netname, dataset_name, dataset_version):
    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, netname)

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

    crop0 = networks.create_crop_fn(1, 0, 1)
    crop1 = networks.create_crop_fn(1, 1, 2)
    crop2 = networks.create_crop_fn(1, 2, 3)
    crop3 = networks.create_crop_fn(1, 3, 4)
    crop4 = networks.create_crop_fn(1, 4, 5)

    fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_3, fmeasure_4, fmeasure_0_1_2_3, fmeasure_0_1_2_3_4,
                 fmeasure_0_2, fmeasure_0_1_2, fmeasure_0_1]

    crops = [crop0, crop1, crop2, crop3, crop4]

    epochs = MAXEPOCHS

    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure

    for crop in crops:
        custom_objects[crop.__name__] = crop

    model_path = os.path.join(save_dir, netname + '_model.json')
    save_weights_only = False
    if os.path.exists(model_path):
        save_weights_only = True

        with open(model_path, "r") as f:
            string = json.load(f)
            model = model_from_json(string, custom_objects=custom_objects)

    for epoch in range(epochs, 0, -1):
        if save_weights_only:
            netpath = os.path.join(save_dir, netname + '_weights.%03d.h5' % epoch)
        else:
            netpath = os.path.join(save_dir, netname + '_completemodel.%03d.h5' % epoch)
        if os.path.exists(netpath):
            last_path = netpath

            print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

            if save_weights_only:
                model.load_weights(last_path)
            else:
                model = load_model(last_path, custom_objects=custom_objects)

            model.summary()

            # TODO: print model details
            """
            details_path = os.path.join(save_dir, netname + 'details.txt')
            print(details_path)
            details_file = open(details_path, 'a')
            details_file.write(str(model))
            details_file.close()
            """

            plot_path = os.path.join(save_dir, netname + 'plot.png')

            plot_model(model, to_file=plot_path, show_shapes=True)

            break



def perform_evaluation(netname, dataset_name, dataset_version, feature_type='bow', context=False, distance=5):

    name = netname
    print(str(time.ctime()) + "\tLAUNCHING TRAINING: " + name)
    print(str(time.ctime()) + "\tLOADING DATASET...")

    dataset, max_text_len, max_prop_len = training.load_dataset(dataset_name=dataset_name,
                                                                dataset_version=dataset_version,
                                                                dataset_split='total',
                                                                feature_type=feature_type,
                                                                min_text_len=2,
                                                                distance=distance,
                                                                context=context)

    print(str(time.ctime()) + "\tDATASET LOADED...")

    sys.stdout.flush()

    print(str(time.ctime()) + "\tPROCESSING DATA AND MODEL...")

    X3_test = []
    X3_train = []
    X3_validation = []
    Y_train = []
    Y_test = []
    Y_validation = []

    split = 'train'
    if len(dataset[split]['source_props']) > 0:
        X_source_train = dataset[split]['source_props']
        del dataset[split]['source_props']
        X_target_train = dataset[split]['target_props']
        del dataset[split]['target_props']
        Y_links_train = np.array(dataset[split]['links'])
        Y_rtype_train = np.array(dataset[split]['relations_type'], dtype=np.float32)
        Y_stype_train = np.array(dataset[split]['sources_type'])
        Y_ttype_train = np.array(dataset[split]['targets_type'])

        # if they are not used, creates a mock (necessary for compatibility with other stuff)
        numdata = len(Y_links_train)
        if distance > 0:
            X_dist_train = dataset[split]['distance']
        else:
            X_dist_train = np.zeros((numdata, 2))
        if context:
            X_marks_train = dataset[split]['mark']
            X_text_train = dataset[split]['texts']
            del dataset[split]['texts']
        else:
            X_marks_train = np.zeros((numdata, 2, 2))
            X_text_train = np.zeros((numdata, 2))

        Y_train = [Y_links_train, Y_rtype_train, Y_stype_train, Y_ttype_train]
        X3_train = [X_text_train, X_source_train, X_target_train, X_dist_train, X_marks_train, ]

        print(str(time.ctime()) + "\t\tTRAINING DATA PROCESSED...")
        print("Length: " + str(len(X3_train[0])))

    split = 'test'

    if len(dataset[split]['source_props']) > 0:
        X_source_test = dataset[split]['source_props']
        del dataset[split]['source_props']
        X_target_test = dataset[split]['target_props']
        del dataset[split]['target_props']
        Y_links_test = np.array(dataset[split]['links'])
        Y_rtype_test = np.array(dataset[split]['relations_type'])
        Y_stype_test = np.array(dataset[split]['sources_type'])
        Y_ttype_test = np.array(dataset[split]['targets_type'])
        Y_test = [Y_links_test, Y_rtype_test, Y_stype_test, Y_ttype_test]

        # if they are not used, creates a mock (necessary for compatibility with other stuff)
        numdata = len(Y_links_test)
        if distance > 0:
            X_dist_test = dataset[split]['distance']
        else:
            X_dist_test = np.zeros((numdata, 2))
        if context:
            X_marks_test = dataset[split]['mark']
            X_text_test = dataset[split]['texts']
            del dataset[split]['texts']
        else:
            X_marks_test = np.zeros((numdata, 2, 2))
            X_text_test = np.zeros((numdata, 2))

        X3_test = [X_text_test, X_source_test, X_target_test, X_dist_test, X_marks_test, ]

        print(str(time.ctime()) + "\t\tTEST DATA PROCESSED...")
        print("Length: " + str(len(X3_test[0])))


    split = 'validation'
    if len(dataset[split]['source_props']) > 0:

        X_source_validation = dataset[split]['source_props']
        del dataset[split]['source_props']
        X_target_validation = dataset[split]['target_props']
        del dataset[split]['target_props']
        Y_links_validation = np.array(dataset[split]['links'])
        Y_rtype_validation = np.array(dataset[split]['relations_type'])
        Y_stype_validation = np.array(dataset[split]['sources_type'])
        Y_ttype_validation = np.array(dataset[split]['targets_type'])
        Y_validation = [Y_links_validation, Y_rtype_validation, Y_stype_validation, Y_ttype_validation]

        # if they are not used, creates a mock (necessary for compatibility with other stuff)
        numdata = len(Y_links_validation)
        if distance > 0:
            X_dist_validation = dataset[split]['distance']
        else:
            X_dist_validation = np.zeros((numdata, 2))
        if context:
            X_text_validation = dataset[split]['texts']
            del dataset[split]['texts']
            X_marks_validation = dataset[split]['mark']
        else:
            X_marks_validation = np.zeros((numdata, 2, 2))
            X_text_validation = np.zeros((numdata, 2))

        X3_validation = [X_text_validation, X_source_validation, X_target_validation, X_dist_validation, X_marks_validation]

        print(str(time.ctime()) + "\t\tVALIDATION DATA PROCESSED...")
        print("Length: " + str(len(X3_validation[0])))

    if context and distance > 0:
        X = {'test': X3_test,
             'train': X3_train,
             'validation': X3_validation}
    elif distance > 0:
        X = {'test': X3_test[1:-1],
             'train': X3_train[1:-1],
             'validation': X3_validation[1:-1]}
    elif context:
        X = {'test': X3_test[:-2] + X3_test[-1:],
             'train': X3_train[:-2] + X3_train[-1:],
             'validation': X3_validation[:-2] + X3_validation[-1:]}
    else:
        X = {'test': X3_test[1:-2],
             'train': X3_train[1:-2],
             'validation': X3_validation[1:-2]}

    Y = {'test': Y_test,
         'train': Y_train,
         'validation': Y_validation}


    # multi-iteration evaluation setting
    final_scores = {'train': [], 'test': [], 'validation': []}
    evaluation_headline = ""
    if dataset_name == "AAEC_v2":
        evaluation_headline = ("set\tF1 AVG all\tF1 AVG LP\tF1 Link\tF1 R AVG dir\tF1 R support\tF1 R attack\t" +
                                "F1 P AVG\tF1 P premise\tF1 P claim\tF1 P major claim\tF1 P avg\n\n")
    elif dataset_name == "cdcp_ACL17":
        evaluation_headline = ("set\tF1 AVG all\tF1 AVG LP\tF1 Link\tF1 R AVG dir\tF1 R reason\tF1 R evidence\t" +
                                "F1 P AVG\t" +
                                "F1 P policy\tF1 P fact\tF1 P testimony\tF1 P value\tF1 P reference\tF1 P avg\n\n")
    elif dataset_name == "RCT":
        evaluation_headline = ("set\tF1 AVG all\tF1 AVG LP\tF1 Link\tF1 R AVG dir\tF1 R support\tF1 R attack\t" +
                                "F1 P AVG\tF1 P premise\tF1 P claim\tF1 P avg" +
                                "Pr P AVG\tPr P premise\tPr P claim\tPr P avg" +
                                "Rec P AVG\tRec P premise\tRec P claim\tRec P avg" +
                                "Fs P AVG\tFs P premise\tFs P claim\tFs P avg" +
                                "Supp P AVG\tSupp P premise\tSupp P claim\tSupp P avg" +
                                "\n\n")

    print(str(time.ctime()) + "\t\tCREATING MODEL...")


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

    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure

    crop0 = networks.create_crop_fn(1, 0, 1)
    crop1 = networks.create_crop_fn(1, 1, 2)
    crop2 = networks.create_crop_fn(1, 2, 3)
    crop3 = networks.create_crop_fn(1, 3, 4)
    crop4 = networks.create_crop_fn(1, 4, 5)

    crops = [crop0, crop1, crop2, crop3, crop4]
    for crop in crops:
        custom_objects[crop.__name__] = crop


    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure

    for crop in crops:
        custom_objects[crop.__name__] = crop


    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)
    model_path = os.path.join(save_dir, netname + '_model.json')

    model = None

    save_weights_only = False
    if os.path.exists(model_path):
        save_weights_only = True

        with open(model_path, "r") as f:
            string = json.load(f)
            model = model_from_json(string, custom_objects=custom_objects)



    iterations = MAXITERATIONS
    file_names = os.listdir(save_dir)
    found = False

    # determine the number of iterations
    for iteration in range(iterations, 0, -1):
        net_file_name = (netname + "_" + iteration + '_weights_')
        for name in file_names:
            if net_file_name in name:
                found = True
                break
        if found:
            iterations = iteration
            break

    # train and test iterations
    for iteration in range(iterations+1):

        # explore all the possible epochs to fine the last one (the first one found)
        last_epoch = MAXEPOCHS
        last_path = ""

        for epoch in range(last_epoch, 0, -1):
            if save_weights_only:
                netpath = os.path.join(save_dir, netname + "_" + iteration + '_weights.%03d.h5' % epoch)
            else:
                netpath = os.path.join(save_dir, netname + "_" + iteration + '_completemodel.%03d.h5' % epoch)
            if os.path.exists(netpath):
                last_path = netpath

                print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

                if save_weights_only:
                    model.load_weights(last_path)
                else:
                    model = load_model(last_path, custom_objects=custom_objects)
                break

        if training.DEBUG:
            plot_model(model, netname + ".png", show_shapes=True)

        print("\n\n\tLOADED NETWORK: " + last_path + "\n")

        testfile = open(os.path.join(save_dir, name + "_eval.txt"), 'w')

        print("\n-----------------------\n")

        testfile.write(evaluation_headline)


        for split in ['test', 'validation', 'train']:

            if len(X[split]) == 0:
                continue

            # 2 dim
            # ax0 = samples
            # ax1 = classes
            Y_pred = model.predict(X[split])

            # begin of the evaluation of the single propositions scores
            sids = dataset[split]['s_id']
            tids = dataset[split]['t_id']

            # dic
            # each values has 2 dim
            # ax0: ids
            # ax1: classes
            s_pred_scores = {}
            s_test_scores = {}
            t_pred_scores = {}
            t_test_scores = {}

            for index in range(len(sids)):
                sid = sids[index]
                tid = tids[index]

                if sid not in s_pred_scores.keys():
                    s_pred_scores[sid] = []
                    s_test_scores[sid] = []
                s_pred_scores[sid].append(Y_pred[2][index])
                s_test_scores[sid].append(Y[split][2][index])

                if tid not in t_pred_scores.keys():
                    t_pred_scores[tid] = []
                    t_test_scores[tid] = []
                t_pred_scores[tid].append(Y_pred[3][index])
                t_test_scores[tid].append(Y[split][3][index])

            Y_pred_prop_real_list = []
            Y_test_prop_real_list = []

            for p_id in t_pred_scores.keys():
                Y_pred_prop_real_list.append(np.concatenate([s_pred_scores[p_id], t_pred_scores[p_id]]))
                Y_test_prop_real_list.append(np.concatenate([s_test_scores[p_id], t_test_scores[p_id]]))

            # 3 dim
            # ax0: ids
            # ax1: samples
            # ax2: classes
            Y_pred_prop_real_list = np.array(Y_pred_prop_real_list)
            Y_test_prop_real_list = np.array(Y_test_prop_real_list)

            Y_pred_prop_real = []
            Y_test_prop_real = []
            for index in range(len(Y_pred_prop_real_list)):
                Y_pred_prop_real.append(np.sum(Y_pred_prop_real_list[index], axis=-2))
                Y_test_prop_real.append(np.sum(Y_test_prop_real_list[index], axis=-2))

            Y_pred_prop_real = np.array(Y_pred_prop_real)
            Y_test_prop_real = np.array(Y_test_prop_real)
            Y_pred_prop_real = np.argmax(Y_pred_prop_real, axis=-1)
            Y_test_prop_real = np.argmax(Y_test_prop_real, axis=-1)
            # end of the evaluation of the single propositions scores

            Y_pred_links = np.argmax(Y_pred[0], axis=-1)
            Y_test_links = np.argmax(Y[split][0], axis=-1)

            Y_pred_rel = np.argmax(Y_pred[1], axis=-1)
            Y_test_rel = np.argmax(Y[split][1], axis=-1)

            # predictions computed! Computing measures!

            # F1s
            score_f1_link = f1_score(Y_test_links, Y_pred_links, average=None, labels=[0])
            score_f1_rel = f1_score(Y_test_rel, Y_pred_rel, average=None, labels=[0, 2])
            score_f1_rel_AVGM = f1_score(Y_test_rel, Y_pred_rel, average='macro', labels=[0, 2])

            score_f1_prop_real = f1_score(Y_test_prop_real, Y_pred_prop_real, average=None)
            score_f1_prop_AVGM_real = f1_score(Y_test_prop_real, Y_pred_prop_real, average='macro')
            score_f1_prop_AVGm_real = f1_score(Y_test_prop_real, Y_pred_prop_real, average='micro')

            score_f1_AVG_LP_real = np.mean([score_f1_link, score_f1_prop_AVGM_real])
            score_f1_AVG_all_real = np.mean([score_f1_link, score_f1_prop_AVGM_real, score_f1_rel_AVGM])

            # Precision-recall-fscore-support
            score_prfs_prop = precision_recall_fscore_support(Y_test_prop_real, Y_pred_prop_real, average=None)
            score_prec_prop = score_prfs_prop[0]
            score_rec_prop = score_prfs_prop[1]
            score_fscore_prop = score_prfs_prop[2]
            score_supp_prop = score_prfs_prop[3]

            score_prfs_prop_AVGM = precision_recall_fscore_support(Y_test_prop_real, Y_pred_prop_real, average='macro')
            score_prec_prop_AVGM = score_prfs_prop_AVGM[0]
            score_rec_prop_AVGM = score_prfs_prop_AVGM[1]
            score_fscore_prop_AVGM = score_prfs_prop_AVGM[2]
            score_supp_prop_AVGM = score_prfs_prop_AVGM[3]

            score_prfs_prop_AVGm = precision_recall_fscore_support(Y_test_prop_real, Y_pred_prop_real, average='micro')
            score_prec_prop_AVGm = score_prfs_prop_AVGm[0]
            score_rec_prop_AVGm = score_prfs_prop_AVGm[1]
            score_fscore_prop_AVGm = score_prfs_prop_AVGm[2]
            score_supp_prop_AVGm = score_prfs_prop_AVGm[3]

            iteration_scores = []
            iteration_scores.append(score_f1_AVG_all_real[0])
            iteration_scores.append(score_f1_AVG_LP_real[0])
            iteration_scores.append(score_f1_link[0])
            iteration_scores.append(score_f1_rel_AVGM)
            for score in score_f1_rel:
                iteration_scores.append(score)
            iteration_scores.append(score_f1_prop_AVGM_real)
            for score in score_f1_prop_real:
                iteration_scores.append(score)
            iteration_scores.append(score_f1_prop_AVGm_real)


            iteration_scores.append(score_prec_prop_AVGM)
            for score in score_prec_prop:
                iteration_scores.append(score)
            iteration_scores.append(score_prec_prop_AVGm)

            iteration_scores.append(score_rec_prop_AVGM)
            for score in score_rec_prop:
                iteration_scores.append(score)
            iteration_scores.append(score_rec_prop_AVGm)

            iteration_scores.append(score_fscore_prop_AVGM)
            for score in score_fscore_prop:
                iteration_scores.append(score)
            iteration_scores.append(score_fscore_prop_AVGm)

            iteration_scores.append(score_supp_prop_AVGM)
            for score in score_supp_prop:
                iteration_scores.append(score)
            iteration_scores.append(score_supp_prop_AVGm)

            final_scores[split].append(iteration_scores)

            # writing single iteration scores
            string = split
            for value in iteration_scores:
                string += "\t" + "{:10.4f}".format(value)

            testfile.write(string + "\n")

            testfile.flush()

        testfile.close()

        # END OF A ITERATION

    # FINAL EVALUATION
    testfile = open(os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version,
                                 netname + "_eval.txt"), 'w')

    testfile.write(evaluation_headline)
    for split in ['test', 'validation', 'train']:
        if len(final_scores[split]) > 0:
            split_scores = np.array(final_scores[split], ndmin=2)
            split_scores = np.average(split_scores, axis=0)

            string = split
            for value in split_scores:
                string += "\t" + "{:10.4f}".format(value)

            testfile.write(string + "\n")

            testfile.flush()
    testfile.close()



def generate_confusion_matrix(netname, dataset_name, dataset_version, feature_type='bow', context=True, distance=True):

    name = netname
    print(str(time.ctime()) + "\tLAUNCHING TRAINING: " + name)
    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = training.load_dataset(dataset_name=dataset_name,
                                                       dataset_version=dataset_version,
                                                       dataset_split='total',
                                                       feature_type=feature_type)
    print(str(time.ctime()) + "\tDATASET LOADED...")

    sys.stdout.flush()

    print(str(time.ctime()) + "\tPROCESSING DATA AND MODEL...")

    split = 'train'
    X_marks_train = dataset[split]['mark']
    X_dist_train = dataset[split]['distance']
    X_text_train = dataset[split]['texts']
    dataset[split]['texts'] = 0
    X_source_train = dataset[split]['source_props']
    dataset[split]['source_props'] = 0
    X_target_train = dataset[split]['target_props']
    dataset[split]['target_props'] = 0
    Y_links_train = np.array(dataset[split]['links'])
    Y_rtype_train = np.array(dataset[split]['relations_type'], dtype=np.float32)
    Y_stype_train = np.array(dataset[split]['sources_type'])
    Y_ttype_train = np.array(dataset[split]['targets_type'])
    X_train = [X_text_train, X_source_train, X_target_train]
    Y_train = [Y_links_train, Y_rtype_train, Y_stype_train, Y_ttype_train]
    X3_train = [X_text_train, X_source_train, X_target_train, X_dist_train, X_marks_train]

    print(str(time.ctime()) + "\t\tTRAINING DATA PROCESSED...")

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

    X_marks_test = dataset[split]['mark']
    X_dist_test = dataset[split]['distance']
    X3_test = [X_text_test, X_source_test, X_target_test, X_dist_test, X_marks_test]


    print(str(time.ctime()) + "\t\tTEST DATA PROCESSED...")

    split = 'validation'
    X_text_validation = dataset[split]['texts']
    dataset[split]['texts'] = 0
    X_source_validation = dataset[split]['source_props']
    dataset[split]['source_props'] = 0
    X_target_validation = dataset[split]['target_props']
    dataset[split]['target_props'] = 0
    Y_links_validation = np.array(dataset[split]['links'])
    Y_rtype_validation = np.array(dataset[split]['relations_type'])
    Y_stype_validation = np.array(dataset[split]['sources_type'])
    Y_ttype_validation = np.array(dataset[split]['targets_type'])
    X_validation = [X_text_validation, X_source_validation, X_target_validation]
    Y_validation = [Y_links_validation, Y_rtype_validation, Y_stype_validation, Y_ttype_validation]

    X_marks_validation = dataset[split]['mark']
    X_dist_validation = dataset[split]['distance']
    X3_validation = [X_text_validation, X_source_validation, X_target_validation, X_dist_validation,
                     X_marks_validation]

    print(str(time.ctime()) + "\t\tVALIDATION DATA PROCESSED...")
    print(str(time.ctime()) + "\t\tCREATING MODEL...")

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

    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure

    crop0 = networks.create_crop_fn(1, 0, 1)
    crop1 = networks.create_crop_fn(1, 1, 2)
    crop2 = networks.create_crop_fn(1, 2, 3)
    crop3 = networks.create_crop_fn(1, 3, 4)
    crop4 = networks.create_crop_fn(1, 4, 5)

    crops = [crop0, crop1, crop2, crop3, crop4]
    for crop in crops:
        custom_objects[crop.__name__] = crop

    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)

    last_epoch = MAXEPOCHS

    last_path = ""

    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure

    for crop in crops:
        custom_objects[crop.__name__] = crop

    model_path = os.path.join(save_dir, netname + '_model.json')

    model = None

    save_weights_only = False
    if os.path.exists(model_path):
        save_weights_only = True

        with open(model_path, "r") as f:
            string = json.load(f)
            model = model_from_json(string, custom_objects=custom_objects)

    for epoch in range(last_epoch, 0, -1):
        if save_weights_only:
            netpath = os.path.join(save_dir, netname + '_weights.%03d.h5' % epoch)
        else:
            netpath = os.path.join(save_dir, netname + '_completemodel.%03d.h5' % epoch)
        if os.path.exists(netpath):
            last_path = netpath

            print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

            if save_weights_only:
                model.load_weights(last_path)
            else:
                model = load_model(last_path, custom_objects=custom_objects)
            break


    print("\n\n\tLOADED NETWORK: " + last_path + "\n")

    if training.DEBUG:
        plot_model(model, netname, show_shapes=True)

    if context and distance:
        X = {'test': X3_test,
             'train': X3_train,
             'validation': X3_validation}
    elif distance:
        X = {'test': X3_test[1:-1],
             'train': X3_train[1:-1],
             'validation': X3_validation[1:-1]}
    elif context:
        X = {'test': X3_test[:-2] + X3_test[-1:],
             'train': X3_train[:-2] + X3_train[-1:],
             'validation': X3_validation[:-2] + X3_validation[-1:]}
    else:
        X = {'test': X3_test[1:-2],
             'train': X3_train[1:-2],
             'validation': X3_validation[1:-2]}

    Y = {'test': Y_test,
         'train': Y_train,
         'validation': Y_validation}

    testfile = open(os.path.join(save_dir, name + "_confusion.txt"), 'w')

    print("\n-----------------------\n")

    for split in ['test', 'validation', 'train']:
        # 2 dim
        # ax0 = samples
        # ax1 = classes
        Y_pred = model.predict(X[split])

        # begin of the evaluation of the single propositions scores
        sids = dataset[split]['s_id']
        tids = dataset[split]['t_id']

        # dic
        # each values has 2 dim
        # ax0: ids
        # ax1: classes
        s_pred_scores = {}
        s_test_scores = {}
        t_pred_scores = {}
        t_test_scores = {}


        for index in range(len(sids)):
            sid = sids[index]
            tid = tids[index]

            if sid not in s_pred_scores.keys():
                s_pred_scores[sid] = []
                s_test_scores[sid] = []
            s_pred_scores[sid].append(Y_pred[2][index])
            s_test_scores[sid].append(Y[split][2][index])

            if tid not in t_pred_scores.keys():
                t_pred_scores[tid] = []
                t_test_scores[tid] = []
            t_pred_scores[tid].append(Y_pred[3][index])
            t_test_scores[tid].append(Y[split][3][index])

        Y_pred_prop_real_list = []
        Y_test_prop_real_list = []

        for p_id in t_pred_scores.keys():
            Y_pred_prop_real_list.append(np.concatenate([s_pred_scores[p_id], t_pred_scores[p_id]]))
            Y_test_prop_real_list.append(np.concatenate([s_test_scores[p_id], t_test_scores[p_id]]))

        # 3 dim
        # ax0: ids
        # ax1: samples
        # ax2: classes
        Y_pred_prop_real_list = np.array(Y_pred_prop_real_list)
        Y_test_prop_real_list = np.array(Y_test_prop_real_list)

        Y_pred_prop_real = []
        Y_test_prop_real = []
        for index in range(len(Y_pred_prop_real_list)):
            Y_pred_prop_real.append(np.sum(Y_pred_prop_real_list[index], axis=-2))
            Y_test_prop_real.append(np.sum(Y_test_prop_real_list[index], axis=-2))


        Y_pred_prop_real = np.array(Y_pred_prop_real)
        Y_test_prop_real = np.array(Y_test_prop_real)
        # end of the evaluation of the single propositions scores


        """
        Y_pred_prop_real = []
        Y_test_prop_real = []
        for index in range(len(Y_pred_prop_real_list)):
            Y_pred_prop_real = np.sum(Y_pred_prop_real_list[index], axis=2)
            Y_test_prop_real = np.sum(Y_test_prop_real_list[index], axis=2)
        """

        # Y_pred_prop = np.concatenate([Y_pred[2], Y_pred[3]])
        # Y_test_prop = np.concatenate([Y[split][2], Y[split][3]])

        Y_pred_prop_real = np.argmax(Y_pred_prop_real, axis=-1)
        Y_test_prop_real = np.argmax(Y_test_prop_real, axis=-1)

        # Y_pred_prop = np.argmax(Y_pred_prop, axis=-1)
        # Y_test_prop = np.argmax(Y_test_prop, axis=-1)

        Y_pred_links = np.argmax(Y_pred[0], axis=-1)
        Y_test_links = np.argmax(Y[split][0], axis=-1)

        Y_pred_rel = np.argmax(Y_pred[1], axis=-1)
        Y_test_rel = np.argmax(Y[split][1], axis=-1)

        confusion_link = confusion_matrix(Y_test_links, Y_pred_links)
        confusion_rel = confusion_matrix(Y_test_rel, Y_pred_rel)
        # confusion_prop = confusion_matrix(Y_test_prop, Y_pred_prop)
        confusion_prop_real = confusion_matrix(Y_test_prop_real, Y_pred_prop_real)

        testfile.write("\n")
        testfile.write(split)
        testfile.write("\n\nlink\n")
        testfile.write(str(confusion_link))
        testfile.write("\n\nrel\n")
        testfile.write(str(confusion_rel))
        # testfile.write("\n\nprop\n")
        # testfile.write(str(confusion_prop))
        testfile.write("\n\nprop real\n")
        testfile.write(str(confusion_prop_real))
        testfile.write("\n\n")

        testfile.flush()
    testfile.close()


if __name__ == '__main__':

    netname = 'prova'
    if len(sys.argv) > 1:
        netname = sys.argv[1]

    name = 'cdcp7R13tv'
    dataset_name = 'cdcp_ACL17'
    dataset_version = 'new_3'
    split = 'test'
    epoch = None
    epoch = 295

    # evaluate_single_epoch(name, dataset_name, dataset_version)

    # evaluate_every_epoch(name, dataset_name, dataset_version)

    # print_model(name, dataset_name, dataset_version)

    perform_evaluation(name, dataset_name, dataset_version)
    generate_confusion_matrix(name, dataset_name, dataset_version)

    name = 'cdcp7R13'
    perform_evaluation(name, dataset_name, dataset_version)
    generate_confusion_matrix(name, dataset_name, dataset_version)
    name = 'cdcp7R13nc'
    perform_evaluation(name, dataset_name, dataset_version, context=False)
    generate_confusion_matrix(name, dataset_name, dataset_version, context=False)
    name = 'cdcp7R13tvnc'
    perform_evaluation(name, dataset_name, dataset_version, context=False)
    generate_confusion_matrix(name, dataset_name, dataset_version, context=False)