__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018-2020 Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.3.0"
__email__ = "a.galassi@unibo.it"

import os
import numpy as np
import sys
import time
import json
import training
import argparse
import krippendorff

from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model, model_from_json
from training_utils import get_avgF1
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from glove_loader import DIM
from scipy import stats
from dataset_config import dataset_info
from networks import (create_sum_fn, create_average_fn,
                      create_count_nonpadding_fn, create_elementwise_division_fn, create_padding_mask_fn,
                      create_mutiply_negative_elements_fn)

import networks


MAXEPOCHS = 1000
MAXITERATIONS = 20


def perform_evaluation(netfolder, dataset_name, dataset_version, feature_type='bow', retrocompatibility=False, distance=5,
                       ensemble=None, ensemble_top_n=1.00, ensemble_top_criterion="link", token_wise=False, error_analysis=False,
                       visualize_attention=False, embed_name="glove300"):
    return_value = 0

    # name of the network
    netname = os.path.basename(netfolder)
    print("Evaluating network: " + str(netname))

    name = netname
    print(str(time.ctime()) + "\tLAUNCHING EVALUATION: " + name)
    print(str(time.ctime()) + "\tLOADING DATASET " + dataset_name)

    this_ds_info = dataset_info[dataset_name]

    output_units = ()
    min_text = 0
    min_prop = 0
    link_as_sum = [[]]

    output_units = this_ds_info["output_units"]
    min_text = this_ds_info["min_text"]
    min_prop = this_ds_info["min_prop"]
    link_as_sum = this_ds_info["link_as_sum"]


    dataset, max_text_len, max_prop_len = training.load_dataset(dataset_name=dataset_name,
                                                                dataset_version=dataset_version,
                                                                dataset_split='total',
                                                                feature_type=feature_type,
                                                                distance=distance,
                                                                min_text_len=min_text,
                                                                min_prop_len=min_prop,
                                                                embed_name=embed_name)


    # for token-wise evaluation, memorize the number of tokens in each proposition
    num_of_tokens = {}
    for split in ['train', 'test', 'validation']:
        for index in range(len(dataset[split]['t_id'])):
            prop_id = dataset[split]['t_id'][index]
            if prop_id not in num_of_tokens.keys():
                target_prop = dataset[split]['target_props'][index]
                num_of_tokens[prop_id] = len(target_prop) - len(np.where(target_prop == 0)[0])

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


        Y_train = [Y_links_train, Y_rtype_train, Y_stype_train, Y_ttype_train]
        if retrocompatibility:
            X_marks_train = np.zeros((numdata, 2, 2))
            X_text_train = np.zeros((numdata, 2))
            X3_train = [X_text_train, X_source_train, X_target_train, X_dist_train, X_marks_train, ]
        else:

            X3_train = [X_source_train, X_target_train, X_dist_train ]

        print(str(time.ctime()) + "\t\tTRAINING DATA PROCESSED...")
        print("Length: " + str(len(X3_train[0])))

    sys.stdout.flush()

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
        if retrocompatibility:
            X_marks_test = np.zeros((numdata, 2, 2))
            X_text_test = np.zeros((numdata, 2))

        if retrocompatibility:
            X_marks_test = np.zeros((numdata, 2, 2))
            X_text_test = np.zeros((numdata, 2))
            X3_test = [X_text_test, X_source_test, X_target_test, X_dist_test, X_marks_test, ]
        else:
            X3_test = [X_source_test, X_target_test, X_dist_test]


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
        if retrocompatibility:
            X_marks_validation = np.zeros((numdata, 2, 2))
            X_text_validation = np.zeros((numdata, 2))

            X3_validation = [X_text_validation, X_source_validation, X_target_validation, X_dist_validation, X_marks_validation]
        else:
            X3_validation = [X_source_validation, X_target_validation, X_dist_validation, ]


        print(str(time.ctime()) + "\t\tVALIDATION DATA PROCESSED...")
        print("Length: " + str(len(X3_validation[0])))


    sys.stdout.flush()

    # multi-iteration evaluation structures
    final_scores = {'train': [], 'test': [], 'validation': []}
    # ensemble structures
    if ensemble is not None and ensemble is not False:
        criterion_scores = []
        ensemble_link_scores = {}
        ensemble_prop_scores = {}
        ensemble_rel_scores = {}
        ensemble_link_votes = {}
        ensemble_prop_votes = {}
        ensemble_rel_votes = {}
        ensemble_link_truth = {}
        ensemble_prop_truth = {}
        ensemble_rel_truth = {}
        for split in ['train', 'test', 'validation']:
            ensemble_link_scores[split] = []
            ensemble_prop_scores[split] = []
            ensemble_rel_scores[split] = []
            ensemble_link_votes[split] = []
            ensemble_prop_votes[split] = []
            ensemble_rel_votes[split] = []
            ensemble_link_truth[split] = []
            ensemble_prop_truth[split] = []
            ensemble_rel_truth[split] = []

    evaluation_headline = this_ds_info["evaluation_headline"]

    print(str(time.ctime()) + "\t\tCREATING MODEL...")

    # used to compute the F1 score for the relations
    relations_labels = this_ds_info["link_as_sum"][0]
    not_a_link_labels = this_ds_info["link_as_sum"][1]


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

    mean_fn = create_average_fn(1)
    sum_fn = create_sum_fn(1)
    division_fn = create_elementwise_division_fn()
    pad_fn = create_count_nonpadding_fn(1, (DIM,))
    padd_fn = create_padding_mask_fn()
    neg_fn = create_mutiply_negative_elements_fn()
    custom_objects[mean_fn.__name__] = mean_fn
    custom_objects[sum_fn.__name__] = sum_fn
    custom_objects[division_fn.__name__] = division_fn
    custom_objects[pad_fn.__name__] = pad_fn
    custom_objects[padd_fn.__name__] = padd_fn
    custom_objects[neg_fn.__name__] = neg_fn


    save_dir = os.path.join(netfolder)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_path = os.path.join(netfolder, netname + '_model.json')

    model = None

    save_weights_only = False
    if os.path.exists(model_path):
        save_weights_only = True

        try:
            with open(model_path, "r") as f:
                string = json.load(f)
                model = model_from_json(string, custom_objects=custom_objects)
        except:
            print("Default model not working, trying alternative")
            model_path = os.path.join(netfolder, netname + '_modelv2.json')
            with open(model_path, "r") as f:
                    string = json.load(f)
                    model = model_from_json(string, custom_objects=custom_objects)

    model.summary()

    iterations = MAXITERATIONS
    file_names = os.listdir(netfolder)
    found = False

    # determine the number of iterations
    for iteration in range(iterations, 0, -1):
        net_file_name = (netname + "_" + str(iteration) + '_weights')
        for name in file_names:
            if net_file_name in name:
                found = True
                break
        if found:
            iterations = iteration
            break

    X = None
    Y = None

    components_id_list = {}
    components_id_list["train"] = []
    components_id_list["test"] = []
    components_id_list["validation"] = []

    def create_report(Y_test_links, Y_pred_links, Y_test_rel, Y_pred_rel, Y_test_prop_real, Y_pred_prop_real, split, error_analysis):

        report = ""

        # F1s
        score_f1_link = f1_score(Y_test_links, Y_pred_links, average=None, labels=[0])
        score_f1_rel = f1_score(Y_test_rel, Y_pred_rel, average=None, labels=relations_labels)
        score_f1_rel_AVGM = f1_score(Y_test_rel, Y_pred_rel, average='macro', labels=relations_labels)
        score_f1_non_link = f1_score(Y_test_links, Y_pred_links, average=None, labels=[1])

        score_f1_rel_completeM = f1_score(Y_test_rel, Y_pred_rel, average='macro')
        score_f1_non_rel = f1_score(Y_test_rel, Y_pred_rel, average=None, labels=[not_a_link_labels[-1]])

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

        score_prfs_prop_AVGM = precision_recall_fscore_support(Y_test_prop_real, Y_pred_prop_real,
                                                               average='macro')
        score_prec_prop_AVGM = score_prfs_prop_AVGM[0]
        score_rec_prop_AVGM = score_prfs_prop_AVGM[1]
        score_fscore_prop_AVGM = score_prfs_prop_AVGM[2]

        score_prfs_prop_AVGm = precision_recall_fscore_support(Y_test_prop_real, Y_pred_prop_real,
                                                               average='micro')
        score_prec_prop_AVGm = score_prfs_prop_AVGm[0]
        score_rec_prop_AVGm = score_prfs_prop_AVGm[1]
        score_fscore_prop_AVGm = score_prfs_prop_AVGm[2]

        iteration_scores = []
        iteration_scores.append(score_f1_AVG_all_real[0])
        iteration_scores.append(score_f1_AVG_LP_real[0])
        iteration_scores.append(score_f1_link[0])

        if split == "validation":
            return_value = score_f1_link[0]

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

        # iteration_scores.append(score_fscore_prop_AVGM)
        # for score in score_fscore_prop:
        #     iteration_scores.append(score)
        # iteration_scores.append(score_fscore_prop_AVGm)

        for score in score_supp_prop:
            iteration_scores.append(score)

        iteration_scores.append(score_f1_non_link[0])

        iteration_scores.append(score_f1_rel_completeM)
        iteration_scores.append(score_f1_non_rel[0])

        final_scores[split].append(iteration_scores)

        # writing single iteration scores
        string = split
        for value in iteration_scores:
            string += "\t" + ("{:10.4f}".format(value)).replace(" ", "")

        print(string)
        shortrep = string

        report += "\n\n"
        report += split

        report += "\nLinks\n"
        report += classification_report(Y_test_links, Y_pred_links,
                                        labels=range(2),
                                        target_names=["YES", "NO"])
        report += "\n"

        report += "\nRelations\n"

        rel_types = []
        for idx in relations_labels:
            rel_type = this_ds_info["rel_types"][idx]
            rel_types.append(rel_type)
            # print("\t" + str(rel_type))
        rel_types.append("None")
        rel_idx = relations_labels.copy()
        rel_idx.append(not_a_link_labels[-1])
        # print(relations_labels)
        # print(rel_idx)
        # print(rel_types)

        report += classification_report(Y_test_rel, Y_pred_rel,
                                        labels=rel_idx,
                                        target_names=rel_types)
        report += "\n"

        report += "\nComponents\n"
        report += classification_report(Y_test_prop_real, Y_pred_prop_real,
                                        labels=range(len(this_ds_info["prop_types"])),
                                        target_names=this_ds_info["prop_types"])
        report += "\n"

        # CONFUSION MATRICES (the code for normalized matrices is in comment cause it requires python
        confusion_link = confusion_matrix(Y_test_links, Y_pred_links)
        # confusion_link2 = confusion_matrix(Y_test_links, Y_pred_links, normalize='true')
        confusion_rel = confusion_matrix(Y_test_rel, Y_pred_rel)
        # confusion_rel2 = confusion_matrix(Y_test_rel, Y_pred_rel, normalize='true')
        confusion_prop_real = confusion_matrix(Y_test_prop_real, Y_pred_prop_real)
        # confusion_prop_real2 = confusion_matrix(Y_test_prop_real, Y_pred_prop_real, normalize='true')
        report += "\n\nlink\n"
        report += str(confusion_link)
        # report += "\n"
        # report += str(confusion_link2)
        report += "\n\nrel\n"
        report += str(confusion_rel)
        # report += "\n"
        # report += str(confusion_rel2)
        report += "\n\nprop real\n"
        report += str(confusion_prop_real)
        # report += "\n"
        # report += str(confusion_prop_real2)
        report += "\n\n"


        if error_analysis:
            # error analysis
            error_analysis_string = ""
            length_corrects = [0] * (max_prop_len + 1)
            length_total = [0] * (max_prop_len + 1)
            length_norm = [0] * (max_prop_len + 1)

            correct_predictions_comp = np.equal(Y_test_prop_real, Y_pred_prop_real)
            for index in range(len(correct_predictions_comp)):
                comp_id = components_id_list[split][index]
                length = num_of_tokens[comp_id]
                if correct_predictions_comp[index]:
                    length_corrects[length] += 1
                length_total[length] += 1

            for index in range(max_prop_len+1):
                if length_total[index] > 0:
                    length_norm[index] = length_corrects[index] / length_total[index]
                else:
                    length_norm[index] = -99
                    length_corrects[index] = -99

            error_analysis_string += "Comp length\t"
            error_analysis_string += "\t".join(map(str, range(max_prop_len + 1)))
            # error_analysis_string += "\ttotal"
            error_analysis_string += "\nComp number\t"
            error_analysis_string += "\t".join(map(str, length_total))
            # error_analysis_string += "\t" + str(sum(length_total))
            error_analysis_string += "\nComp correctness\t"
            error_analysis_string += "\t".join(map(str, length_corrects))
            # error_analysis_string += "\t" + str(sum(length_corrects))
            error_analysis_string += "\nComp norm\t"
            error_analysis_string += "\t".join(map(str, length_norm))

            correct_predictions_link = np.equal(Y_test_links, Y_pred_links)
            truelink = np.equal(Y_test_links, [0]*len(Y_test_links))
            difference_corrects = [0] * (distance*2 + 1)
            difference_total = [0] * (distance*2 + 1)
            difference_norm = [0] * (distance*2 + 1)
            differences = dataset[split]['difference']

            for index in range(len(correct_predictions_link)):
                difference = differences[index]
                if difference > distance:
                    difference = distance
                elif difference < -distance:
                    difference = -distance
                difference += 5
                if truelink[index]:
                    if correct_predictions_link[index]:
                        difference_corrects[difference] += 1
                    difference_total[difference] += 1

            for index in range(distance*2 + 1):
                if difference_total[index] > 0:
                    difference_norm[index] = difference_corrects[index] / difference_total[index]
                else:
                    difference_norm[index] = -99
                    difference_corrects[index] = -99

            error_analysis_string += "\n\nLink distance\t"
            error_analysis_string += "\t".join(map(str, range(-distance, distance+1)))
            # error_analysis_string += "\ttotal"
            error_analysis_string += "\nLink number\t"
            error_analysis_string += "\t".join(map(str, difference_total))
            # error_analysis_string += "\t" + str(sum(difference_total))
            error_analysis_string += "\nLink correctness\t"
            error_analysis_string += "\t".join(map(str, difference_corrects))
            # error_analysis_string += "\t" + str(sum(difference_corrects))
            error_analysis_string += "\nLink norm\t"
            error_analysis_string += "\t".join(map(str, difference_norm))
            error_analysis_string += "\n\n"

            report += error_analysis_string

        return shortrep, report

    # train and test iterations
    for iteration in range(iterations+1):

        print("Evaluating networks: " + str(iteration+1) + "/" + str(iterations+1))
        sys.stdout.flush()

        # explore all the possible epochs to fine the last one (the first one found)
        last_epoch = MAXEPOCHS
        last_path = ""

        for epoch in range(last_epoch, 0, -1):
            if save_weights_only:
                netpath = os.path.join(netfolder, netname + "_" + str(iteration) + '_weights.%03d.h5' % epoch)
            else:
                netpath = os.path.join(netfolder, netname + "_" + str(iteration) + '_completemodel.%03d.h5' % epoch)

            if os.path.exists(netpath):
                last_path = netpath

                print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

                if save_weights_only:
                    model.load_weights(last_path)
                else:
                    model = load_model(last_path, custom_objects=custom_objects)
                break


        if X == None:
            if not distance:
                X = {'test': X3_test[0:-2],
                     'train': X3_train[0:-2],
                     'validation': X3_validation[0:-2]}
            else:
                X = {'test': X3_test,
                     'train': X3_train,
                     'validation': X3_validation}

            Y = {'test': Y_test,
                 'train': Y_train,
                 'validation': Y_validation}

        if training.DEBUG:
            plot_model(model, netname + ".png", show_shapes=True)

        if last_path == "":
            print("ERROR! NO NETWORK LOADED!\n\tExpected example of network name: " + str(netpath))
            exit(1)

        print("\n\n\tLOADED NETWORK: " + last_path + "\n")

        testfile = open(os.path.join(netfolder, netname + "_" + str(iteration) + "_eval.txt"), 'a')

        testfile.write("\n\n")
        testfile.write("DATASET VERSION:\n")
        testfile.write(dataset_version)
        testfile.write("\n")

        print("\n-----------------------\n")

        if token_wise:
            testfile.write("TOKEN-WISE EVALUATION")
        else:
            testfile.write("COMPONENT-WISE EVALUATION")
        testfile.write("\n")

        testfile.write(evaluation_headline)

        print(evaluation_headline)

        extensive_report = ""

        for split in ['test', 'validation', 'train']:

            # TODO: see if this works!
            if visualize_attention:

                dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
                vocabulary_path = os.path.join(dataset_path, 'resources', embed_name, dataset_version,
                                               'glove.embeddings.npz')
                if not os.path.exists(vocabulary_path):
                    vocabulary_path = os.path.join(dataset_path, 'resources', embed_name, 'glove.embeddings.npz')
                vocabulary_list = np.load(vocabulary_path)
                embed_list = vocabulary_list['embeds']
                word_list = vocabulary_list['vocab']

                source_attention_layer = model.get_layer("att_weights_reshape_source")
                target_attention_layer = model.get_layer("att_weights_reshape_target")
                source_scores = source_attention_layer(X[split][0:1])
                target_scores = target_attention_layer(X[split][0:1])

                source_sentences = []
                target_sentences = []

                for instance in X[split][0:1][0]:
                    sentence = []
                    for index in instance:
                        word = word_list[index]
                        sentence.append(word)
                    source_sentences.append(sentence)

                for instance in X[split][0:1][1]:
                    sentence = []
                    for index in instance:
                        word = word_list[index]
                        sentence.append(word)
                    target_sentences.append(sentence)

                for index in range(len(source_sentences)):
                    print(str(source_sentences[index]))
                    print(str(source_scores[index]))
                    print(str(target_sentences[index]))
                    print(str(target_scores[index]))
                exit()


            if len(X[split][0]) <= 1:
                continue

            # 2 dim
            # ax0 = samples
            # ax1 = classes

            Y_pred = model.predict(X[split])

            # every proposition is evaluated multiple times. all these evaluation must be merged together.
            # merging is performed choosing the class that has received the highest probability score summing all the cases
            # it is equivalent to the class that has received the highest probability on average
            # Possible alternative: voting

            # --- begin of the evaluation of the single propositions scores
            sids = dataset[split]['s_id']  # list of source_ids for each pair
            tids = dataset[split]['t_id']  # list of target_ids for each pair

            # dictionaries that will contain the predictions and the truth for each component
            # each values has 2 dim
            # ax0: ids
            # ax1: classes
            s_pred_scores = {}
            s_test_scores = {}
            t_pred_scores = {}
            t_test_scores = {}

            # for each component, gather the list of predictions and ground truth (keeps separate source and target)
            for index in range(len(sids)):
                sid = sids[index]
                tid = tids[index]

                if sid not in s_pred_scores.keys():  # id not seen before
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

            if len(components_id_list[split]) == 0:
                for p_id in sorted(t_pred_scores.keys()):
                    components_id_list[split].append(p_id)

            # merges sources and targets
            for p_id in sorted(t_pred_scores.keys()):
                Y_pred_prop_real_list.append(np.concatenate([s_pred_scores[p_id], t_pred_scores[p_id]]))
                Y_test_prop_real_list.append(np.concatenate([s_test_scores[p_id], t_test_scores[p_id]]))

            # 3 dim
            # ax0: ids
            # ax1: samples
            # ax2: classes
            # value: score associated to the class
            Y_pred_prop_real_list = np.array(Y_pred_prop_real_list)
            Y_test_prop_real_list = np.array(Y_test_prop_real_list)

            Y_pred_prop_real = []
            Y_test_prop_real = []
            # for each id, sum the prediction scores for each class across the samples
            for index in range(len(Y_pred_prop_real_list)):
                Y_pred_prop_real.append(np.sum(Y_pred_prop_real_list[index], axis=-2))
                Y_test_prop_real.append(np.sum(Y_test_prop_real_list[index], axis=-2))

            # 2 dim: ids, classes; value: score for each class
            Y_pred_scores_prop_real = np.array(Y_pred_prop_real)
            Y_test_scores_prop_real = np.array(Y_test_prop_real)


            # select the class that has received the highest probability
            Y_pred_prop_real = np.argmax(Y_pred_scores_prop_real, axis=-1)
            Y_test_prop_real = np.argmax(Y_test_scores_prop_real, axis=-1)

            # --- end of the evaluation of the single propositions scores

            Y_pred_links = np.argmax(Y_pred[0], axis=-1)
            Y_test_links = np.argmax(Y[split][0], axis=-1)

            Y_pred_rel = np.argmax(Y_pred[1], axis=-1)
            Y_test_rel = np.argmax(Y[split][1], axis=-1)


            # If a comparison with a sequence tagging method is needed, it is necessary to split components into tokens
            if token_wise:
                print("Splitting token for token-wise")
                # test and predictions are sorted according to components_id_list

                if len(components_id_list[split]) != len(Y_pred_prop_real):
                    print(len(components_id_list[split]))
                    print(len(Y_pred_prop_real))
                    print("ERROR!!! Components_id_list must be as long as the list of the predictions!!!")
                    sys.exit(40)

                Y_pred_scores_tok_real = []
                Y_pred_tok_real = []
                Y_test_tok_real = []

                # expand each element for the number of tokens
                for index in range(len(components_id_list[split])):
                    id_prop = components_id_list[split][index]
                    prop_pred = Y_pred_prop_real[index]
                    prop_test = Y_test_prop_real[index]
                    prop_score = Y_pred_scores_prop_real[index]
                    tokens = num_of_tokens[id_prop]
                    for j in range(tokens):
                        Y_pred_scores_tok_real.append(prop_score)
                        Y_pred_tok_real.append(prop_pred)
                        Y_test_tok_real.append(prop_test)

                # overwrite previous arrays
                Y_pred_scores_prop_real = np.array(Y_pred_scores_tok_real)
                Y_pred_prop_real = np.array(Y_pred_tok_real)
                Y_test_prop_real = np.array(Y_test_tok_real)


            Y_pred_scores_links = Y_pred[0]
            Y_pred_scores_rel = Y_pred[1]

            # Remove all the symmetric links
            reflexive = []
            for index in range(len(sids)):
                sid = sids[index]
                tid = tids[index]
                if sid == tid:
                    reflexive.append(index)
            if len(reflexive) > 0:
                Y_test_rel = np.delete(Y_test_rel, reflexive, axis=0)
                Y_pred_rel = np.delete(Y_pred_rel, reflexive, axis=0)
                Y_test_links = np.delete(Y_test_links, reflexive, axis=0)
                Y_pred_links = np.delete(Y_pred_links, reflexive, axis=0)
                Y_pred_scores_links = np.delete(Y_pred_scores_links, reflexive, axis=0)
                Y_pred_scores_rel = np.delete(Y_pred_scores_rel, reflexive, axis=0)

            # Merge all the not-link labels in the first of them
            for label in not_a_link_labels:
                Y_test_rel = np.where(Y_test_rel == label, not_a_link_labels[-1], Y_test_rel)
                Y_pred_rel = np.where(Y_pred_rel == label, not_a_link_labels[-1], Y_pred_rel)


            if ensemble is not None and ensemble is not False:
                ensemble_prop_truth[split] = Y_test_prop_real
                ensemble_link_truth[split] = Y_test_links
                ensemble_rel_truth[split] = Y_test_rel

                ensemble_prop_scores[split].append(Y_pred_scores_prop_real)
                ensemble_prop_votes[split].append(Y_pred_prop_real)

                ensemble_link_scores[split].append(Y_pred_scores_links)
                ensemble_link_votes[split].append(Y_pred_links)

                ensemble_rel_scores[split].append(Y_pred_scores_rel)
                ensemble_rel_votes[split].append(Y_pred_rel)

            # predictions computed! Computing measures!

            shortrep, report = create_report(Y_test_links, Y_pred_links, Y_test_rel, Y_pred_rel, Y_test_prop_real, Y_pred_prop_real, split, error_analysis)
            extensive_report += report

            testfile.write(shortrep)
            testfile.write("\n")

            testfile.flush()
            sys.stdout.flush()

        testfile.write(extensive_report)

        print("------------------\n\n------------------\n\n")
        testfile.write("------------------\n\n------------------\n\n")
        testfile.close()

        # END OF A ITERATION

    print(str(time.ctime()) + "\tFINAL EVALUATIONS")

    # FINAL EVALUATION
    file_folder = os.path.join(netfolder)
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    testfile = open(os.path.join(file_folder, os.path.pardir, netname + "_eval.txt"), 'a')

    testfile.write("\n\n")
    testfile.write("DATASET VERSION:\n")
    testfile.write(dataset_version)
    testfile.write("\n")

    print("\n-----------------------\n")

    if token_wise:
        testfile.write("TOKEN-WISE EVALUATION")
    else:
        testfile.write("COMPONENT-WISE EVALUATION")
    testfile.write("\n")

    testfile.write(evaluation_headline)
    print(evaluation_headline)
    extensive_report = ""

    for split in ['test', 'validation', 'train']:
        if len(final_scores[split]) > 0:
            split_scores = np.array(final_scores[split], ndmin=2)
            split_scores = np.average(split_scores, axis=0)

            string = split
            for value in split_scores:
                string += "\t" + ("{:10.4f}".format(value)).replace(" ", "")

            testfile.write(string + "\n")
            print(string)

            sys.stdout.flush()
            testfile.flush()

    # ENSEMBLE SCORE
    # TODO: implement ensemble score consideration
    # REMEMBER THAT not-link relation votes have been merged, but the scores have not
    print(str(time.ctime()) + "\t\tENSEMBLE EVALUATION")
    if ensemble is not None and ensemble is not False:

        testfile.write("\n\n")
        testfile.write(dataset_version)

        # take only the top N networks (according to validation) into account
        ensemble_top_n = int((iterations+1) * ensemble_top_n)
        if ensemble_top_n < (iterations+1):
            testfile.write("\tENSEMBLE\t" + "top " + str(ensemble_top_n) + "/" + str(iterations+1) + "\t"
                           + ensemble_top_criterion + "\n")
            top_n_indexes = []

            # find the indexes of the top N networks
            print("The scores are: " + str(criterion_scores))
            for i in range(0, ensemble_top_n):
                index = np.argmax(criterion_scores)
                top_n_indexes.append(index)
                criterion_scores[index] = 0

            print("The top networks are: " + str(top_n_indexes))

            # create the arrays with only the top k networks votes
            for split in ['test', 'validation', 'train']:
                new_link_votes = []
                new_prop_votes = []
                new_rel_votes = []
                for i in range(0, len(top_n_indexes)):
                    top_index = top_n_indexes[i]
                    new_link_votes.append(ensemble_link_votes[split][top_index])
                    new_prop_votes.append(ensemble_prop_votes[split][top_index])
                    new_rel_votes.append(ensemble_rel_votes[split][top_index])

                ensemble_link_votes[split] = new_link_votes
                ensemble_prop_votes[split] = new_prop_votes
                ensemble_rel_votes[split] = new_rel_votes
        else:
            testfile.write("\tENSEMBLE\t/" + str(iterations+1) + "\n")

        testfile.write(evaluation_headline)
        print(evaluation_headline)

        for split in ['test', 'validation', 'train']:
            if len(final_scores[split]) > 0:

                # compute the answer of the ensemble as the mode of the predictions
                prop_votes = np.array(ensemble_prop_votes[split])
                # print(prop_votes)
                Y_pred_prop_real = stats.mode(prop_votes)[0][0]
                # print(Y_pred_prop_real)
                Y_test_prop_real = ensemble_prop_truth[split]
                # print(Y_test_prop_real)


                # compute IAA between networks: krippendorff's alpha
                kalpha_prop = krippendorff.alpha(reliability_data=ensemble_prop_votes[split],
                                                 value_domain=range(this_ds_info['output_units'][2]))
                kalpha_link = krippendorff.alpha(reliability_data=ensemble_link_votes[split],
                                                 value_domain=[0, 1])
                kalpha_rel = krippendorff.alpha(reliability_data=ensemble_rel_votes[split],
                                                 value_domain=range(this_ds_info['output_units'][1]))

                link_votes = np.array(ensemble_link_votes[split])
                Y_pred_links = stats.mode(link_votes)[0][0]
                Y_test_links = ensemble_link_truth[split]

                rel_votes = np.array(ensemble_rel_votes[split])
                Y_pred_rel = stats.mode(rel_votes)[0][0]
                Y_test_rel = ensemble_rel_truth[split]

                shortrep, report = create_report(Y_test_links, Y_pred_links, Y_test_rel, Y_pred_rel, Y_test_prop_real,
                                                 Y_pred_prop_real, split, error_analysis)


                report += "\n\nkrippendorff's alpha\n"
                report += "Propositions:\t" + str(kalpha_prop) + "\n"
                report += "Link:\t" + str(kalpha_link) + "\n"
                report += "Relations:\t" + str(kalpha_rel) + "\n"


                extensive_report += report

                testfile.write(shortrep)
                testfile.write("\n")

                testfile.flush()
                sys.stdout.flush()


        print("------------------\n\n------------------\n\n")

    testfile.write(extensive_report)
    testfile.write("------------------------------------------------------------------------------------------\n")
    testfile.write("------------------------------------------------------------------------------------------\n")
    testfile.write("------------------------------------------------------------------------------------------\n\n\n\n")
    testfile.close()

    # return the F1
    return return_value




def regenerate_network(netname="cdcp7net2018", dataset_name='cdcp_ACL17', dataset_version='new_3', split='total', netnum=7):
    """
    In case of impossibility to load old networks model, try to recreate it with code and save the new model.
    e.g. problems between python 3.5 and python 3.6
    :param netnum: type of network. Only 7 and 11 are supported (ResArg and ResAttArg). In case of different networks,
    use the related .info file and past here the parameters
    :return:
    """
    iterations=1


    # training config:
    if netnum==7:
        save_weights_only=True
        epochs=10000
        feature_type='bow'
        patience=200
        loss_weights=[0, 10, 1, 1]
        lr_alfa=0.005
        lr_kappa=0.001
        beta_1=0.9
        beta_2=0.9999
        res_scale=60 # res_siz =5
        resnet_layers=(1, 2)
        embedding_scale=6# embedding_size=50
        embedder_layers=4
        final_scale=15 # final_size=20
        space_scale=10
        batch_size=500,
        regularizer_weight=0.0001
        dropout_resnet=0.1
        dropout_embedder=0.1
        dropout_final=0.1
        bn_embed=True
        bn_res=True
        bn_final=True
        network=7
        monitor="links"
        true_validation=True
        temporalBN=False
        same_layers=False
        distance=5
        merge=None
        single_LSTM=True
        pooling=10
        text_pooling=50
        pooling_type='avg'
        classification="softmax"
        dataset_name=dataset_name
        dataset_version=dataset_version
        dataset_split=split
        distance_train_limit=-1
    elif netnum== 11:
        save_weights_only=True,
        epochs=10000
        feature_type='bow'
        patience=100
        loss_weights=[0, 10, 1, 1]
        lr_alfa=0.005
        lr_kappa=0.001
        beta_1=0.9
        beta_2=0.9999
        res_scale=60 # res_siz =5
        resnet_layers=(1, 2)
        embedding_scale=6 # embedding_size=50
        embedder_layers=4
        final_scale=15 # final_size=20
        space_scale=10
        batch_size=500
        regularizer_weight=0.0001
        dropout_resnet=0.1
        dropout_embedder=0.1
        dropout_final=0.1
        bn_embed=True
        bn_res=True
        bn_final=True
        network=11
        monitor="links"
        true_validation=True
        temporalBN=False
        same_layers=False
        distance=5
        merge=None
        single_LSTM=True
        pooling=10
        text_pooling=50
        pooling_type='avg'
        dataset_name=dataset_name
        dataset_version=dataset_version
        dataset_split=split
        distance_train_limit=-1


    if True:
        embedding_size = int(DIM / embedding_scale)
        res_size = int(DIM / res_scale)
        final_size = int(DIM / final_scale)

        parameters = locals()

        save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        output_units = ()
        min_text = 0
        min_prop = 0
        link_as_sum = [[]]

        output_units = dataset_info[dataset_name]["output_units"]
        min_text = dataset_info[dataset_name]["min_text"]
        min_prop = dataset_info[dataset_name]["min_prop"]
        link_as_sum = dataset_info[dataset_name]["link_as_sum"]

        distance_num = distance
        if distance < 0:
            distance = False
        else:
            distance = True

        print(str(time.ctime()) + "\tLOADING DATASET...")
        dataset, max_text_len, max_prop_len = training.load_dataset(dataset_name=dataset_name,
                                                           dataset_version=dataset_version,
                                                           dataset_split=dataset_split,
                                                           feature_type=feature_type,
                                                           min_text_len=min_text,
                                                           min_prop_len=min_prop,
                                                           distance=distance_num,
                                                           distance_train_limit=distance_train_limit,
                                                                    embed_name="glove300")
        print(str(time.ctime()) + "\tDATASET LOADED...")
        sys.stdout.flush()

        print(str(time.ctime()) + "\tPROCESSING DATA AND MODEL...")

        split = 'train'
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

        Y_train = [Y_links_train, Y_rtype_train, Y_stype_train, Y_ttype_train]
        X3_train = [X_source_train, X_target_train, X_dist_train, ]

        print(str(time.ctime()) + "\t\tTRAINING DATA PROCESSED...")
        print("Length: " + str(len(X3_train[0])))

        split = 'test'

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

        X3_test = [X_source_test, X_target_test, X_dist_test]

        print(str(time.ctime()) + "\t\tTEST DATA PROCESSED...")
        print("Length: " + str(len(X3_test[0])))

        split = 'validation'
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

        X3_validation = [X_source_validation, X_target_validation, X_dist_validation, ]

        print(str(time.ctime()) + "\t\tVALIDATION DATA PROCESSED...")
        print("Length: " + str(len(X3_validation[0])))
        print(str(time.ctime()) + "\t\tCREATING MODEL...")

        bow = None
        if feature_type == 'bow':
            dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
            vocabulary_path = os.path.join(dataset_path, 'resources', "glove300", dataset_version,
                                           'glove.embeddings.npz')
            if not os.path.exists(vocabulary_path):
                vocabulary_path = os.path.join(dataset_path, 'resources', "glove300", 'glove.embeddings.npz')
            vocabulary_list = np.load(vocabulary_path)
            embed_list = vocabulary_list['embeds']
            word_list = vocabulary_list['vocab']

            bow = np.zeros((len(word_list) + 1, DIM))
            for index in range(len(word_list)):
                bow[index + 1] = embed_list[index]
            print(str(time.ctime()) + "\t\t\tEMBEDDINGS LOADED...")

        realname = netname

        # multi-iteration evaluation setting
        final_scores = {'train': [], 'test': [], 'validation': []}
        evaluation_headline = ""
        evaluation_headline = dataset_info[dataset_name]["evaluation_headline_short"]

        save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, realname)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if netnum ==7:
        model = networks.build_net_7(bow=bow,
                        link_as_sum=link_as_sum,
                        propos_length=max_prop_len,
                        regularizer_weight=regularizer_weight,
                        dropout_embedder=dropout_embedder,
                        dropout_resnet=dropout_resnet,
                        embedding_size=embedding_size,
                        embedder_layers=embedder_layers,
                        resnet_layers=resnet_layers,
                        res_size=res_size,
                        final_size=final_size,
                        outputs=output_units,
                        bn_embed=bn_embed,
                        bn_res=bn_res,
                        bn_final=bn_final,
                        single_LSTM=single_LSTM,
                        pooling=pooling,
                        text_pooling=text_pooling,
                        pooling_type=pooling_type,
                        dropout_final=dropout_final,
                        same_DE_layers=same_layers,
                        distance=distance_num,
                        temporalBN=temporalBN,)
    elif netnum== 11:
        model = networks.build_net_11(bow=bow,
                             link_as_sum=link_as_sum,
                             propos_length=max_prop_len,
                             regularizer_weight=regularizer_weight,
                             dropout_embedder=dropout_embedder,
                             dropout_resnet=dropout_resnet,
                             embedding_size=embedding_size,
                             embedder_layers=embedder_layers,
                             resnet_layers=resnet_layers,
                             res_size=res_size,
                             final_size=final_size,
                             outputs=output_units,
                             bn_embed=bn_embed,
                             bn_res=bn_res,
                             bn_final=bn_final,
                             single_LSTM=single_LSTM,
                             dropout_final=dropout_final,
                             same_DE_layers=same_layers,
                             distance=distance_num,
                             temporalBN=temporalBN, )


    netfolder = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, netname)

    for iteration in range(iterations+1):

        print("Evaluating networks: " + str(iteration+1) + "/" + str(iterations+1))
        sys.stdout.flush()

        # explore all the possible epochs to fine the last one (the first one found)
        last_epoch = MAXEPOCHS
        last_path = ""

        for epoch in range(last_epoch, 0, -1):
            netpath = os.path.join(netfolder, netname + "_" + str(iteration) + '_weights.%03d.h5' % epoch)

            print(netpath)

            if os.path.exists(netpath):
                last_path = netpath

                print(str(time.ctime()) + "\tLOADING NETWORK: " + last_path)

                if save_weights_only:
                    model.load_weights(last_path)

                    print(" YEEEESSSSSSSSSSSSS")

                    model_name = netname + '_modelv2.json'
                    json_model = model.to_json()
                    with open(os.path.join(save_dir, model_name), 'w') as outfile:
                        json.dump(json_model, outfile)

                    exit()
    exit()



def RCT_routine(netname="RCT11", retrocompatibility=False, distance=5, ensemble=True, token_wise=True, error_analysis=False):

    dataset_name = "RCT"
    training_dataset_version = "neo"

    netpath = os.path.join(os.getcwd(), 'network_models', dataset_name, training_dataset_version, netname)

    test_dataset_version = "neo"

    perform_evaluation(netpath, dataset_name, test_dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=token_wise, error_analysis=error_analysis)

    test_dataset_version = "mixed"

    perform_evaluation(netpath, dataset_name, test_dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=token_wise, error_analysis=error_analysis)

    test_dataset_version = "glaucoma"

    perform_evaluation(netpath, dataset_name, test_dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=token_wise, error_analysis=error_analysis)


def drinv_routine(netname="RCT11", retrocompatibility=False, distance=5, ensemble=True, token_wise=True, error_analysis=False):

    dataset_name = 'DrInventor'
    dataset_version = 'arg10'
    netname = netname

    netpath = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, netname)

    perform_evaluation(netpath, dataset_name, dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=token_wise, error_analysis=error_analysis)


def ECHR_routine(netname, retrocompatibility=False, distance=5, ensemble=True, token_wise=False, error_analysis=False):

    dataset_name = 'ECHR2018'
    dataset_version = 'arg0'
    netname = netname

    netpath = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, netname)

    perform_evaluation(netpath, dataset_name, dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=token_wise, error_analysis=error_analysis)




def cdcp_routine(netname='cdcp111', retrocompatibility=False, distance=5, ensemble=True, token_wise=False, error_analysis=False):

    dataset_name = 'cdcp_ACL17'
    training_dataset_version = 'new_3'
    test_dataset_version = "new_3"
    netname = netname

    netpath = os.path.join(os.getcwd(), 'network_models', dataset_name, training_dataset_version, netname)

    perform_evaluation(netpath, dataset_name, test_dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=token_wise, error_analysis=error_analysis)
    # perform_evaluation(netpath, dataset_name, test_dataset_version, context=False, distance=5,
    #                    ensemble=True, ensemble_top_criterion="link", ensemble_top_n=0.3)


def UKP_routine(netname, retrocompatibility=False, distance=5, ensemble=True, token_wise=False, error_analysis=False):

    dataset_name = 'AAEC_v2'
    training_dataset_version = 'new_2'
    test_dataset_version = "new_2"
    netname = netname

    netpath = os.path.join(os.getcwd(), 'network_models', dataset_name, training_dataset_version, netname)

    perform_evaluation(netpath, dataset_name, test_dataset_version, retrocompatibility=retrocompatibility,
                       distance=distance, ensemble=ensemble, token_wise=token_wise)
    perform_evaluation(netpath, dataset_name, test_dataset_version, retrocompatibility=retrocompatibility, distance=distance, ensemble=ensemble, token_wise=True, error_analysis=error_analysis)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluate a neural network approach.")
    parser.add_argument("netname", help="The name of the network")


    parser.add_argument('-c', '--corpus',
                        choices=["rct", "drinv", "cdcp", "echr", "ukp"],
                        help="corpus", default="cdcp")
    parser.add_argument('-r', '--retrocompatibility', help="Use of a network trained with the previous approach", action="store_true")
    parser.add_argument('-d', '--distance', help="The maximum distance considered in the features", default=5)
    parser.add_argument('-e', '--ensemble', help="Use of an ensemble of networks", action="store_true")
    parser.add_argument('-t', '--token', help="Perform token-wise component classification (instead of component-wise)", action="store_true")
    parser.add_argument('-x', '--default', help="Perform the evaluation on the dataset with the default options configured for that specific dataset", action="store_true")
    parser.add_argument('-a', '--analysis', help="Perform error analysis", action="store_true")

    args = parser.parse_args()

    netname = args.netname
    corpus = args.corpus
    distance = args.distance
    ensemble = args.ensemble
    retrocompatibility = args.retrocompatibility
    token_wise = args.token
    error_analysis = args.analysis
    default = args.default

    if default:
        if corpus.lower() == "rct":
            RCT_routine(netname)
        elif corpus.lower() == "cdcp":
            cdcp_routine(netname)
        elif corpus.lower() == "drinv":
            drinv_routine(netname)
        elif corpus.lower() == "ukp":
            UKP_routine(netname)
    else:
        if corpus.lower() == "rct":
            RCT_routine(netname, retrocompatibility, distance, ensemble, token_wise, error_analysis)
        elif corpus.lower() == "cdcp":
            cdcp_routine(netname, retrocompatibility, distance, ensemble, token_wise, error_analysis)
        elif corpus.lower() == "drinv":
            drinv_routine(netname, retrocompatibility, distance, ensemble, token_wise, error_analysis)
        elif corpus.lower() == "ukp":
            UKP_routine(netname, retrocompatibility, distance, ensemble, token_wise, error_analysis)



