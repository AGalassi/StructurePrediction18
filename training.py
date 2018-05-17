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
import random
import evaluate_net
import json


from networks import build_net_1, build_net_2, build_net_4, build_net_5, build_net_7, create_crop_fn
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model, model_from_json
from keras.preprocessing.sequence import  pad_sequences
from training_utils import TimingCallback, create_lr_annealing_function, fmeasure, get_avgF1, RealValidationCallback
from glove_loader import DIM
from sklearn.metrics import f1_score

DEBUG = False

def load_dataset(dataset_split='total', dataset_name='cdcp_ACL17', dataset_version='new_2',
                 feature_type='embeddings', min_text_len=0, min_prop_len=0, distance=5):

    max_prop_len = min_prop_len
    max_text_len = min_text_len

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, dataset_split + '.pkl')
    embed_path = os.path.join(dataset_path, feature_type, dataset_version)

    df = pandas.read_pickle(dataframe_path)

    if dataset_name=='cdcp_ACL17':
        categorical_prop = {'policy': [1, 0, 0, 0, 0],
                            'fact': [0, 1, 0, 0, 0],
                            'testimony': [0, 0, 1, 0, 0],
                            'value': [0, 0, 0, 1, 0],
                            'reference': [0, 0, 0, 0, 1],
                            }

        categorical_link = {'reasons': [1, 0, 0, 0, 0],
                            'inv_reasons': [0, 1, 0, 0, 0],
                            'evidences': [0, 0, 1, 0, 0],
                            'inv_evidences': [0, 0, 0, 1, 0],
                            None: [0, 0, 0, 0, 1],
                            }
                            
    elif dataset_name=='AAEC_v2':
        categorical_prop = {'Premise': [1, 0, 0,],
                            'Claim': [0, 1, 0,],
                            'MajorClaim': [0, 0, 1],
                            }

        categorical_link = {'supports': [1, 0, 0, 0, 0],
                            'inv_supports': [0, 1, 0, 0, 0],
                            'attacks': [0, 0, 1, 0, 0],
                            'inv_attacks': [0, 0, 0, 1, 0],
                            None: [0, 0, 0, 0, 1],
                            }

    dataset = {}

    for split in ('train', 'validation', 'test'):
        dataset[split] = {}
        dataset[split]['texts'] = []
        dataset[split]['source_props'] = []
        dataset[split]['target_props'] = []
        dataset[split]['links'] = []
        dataset[split]['relations_type'] = []
        dataset[split]['sources_type'] = []
        dataset[split]['targets_type'] = []
        dataset[split]['distance'] = []
        dataset[split]['mark'] = []
        dataset[split]['s_id'] = []
        dataset[split]['t_id'] = []

    for index, row in df.iterrows():

        text_ID = row['text_ID']
        source_ID = row['source_ID']
        target_ID = row['target_ID']
        split = row['set']


        if row['source_to_target']:
            dataset[split]['links'].append([1, 0])
        else:
            dataset[split]['links'].append([0, 1])
        """
        else:
            if split == 'train':
                n = random.random()
                if n < 0.2:
                    continue
            dataset[split]['links'].append([0, 1])
        """
        
        dataset[split]['sources_type'].append(categorical_prop[row['source_type']])
        dataset[split]['targets_type'].append(categorical_prop[row['target_type']])
        dataset[split]['relations_type'].append(categorical_link[row['relation_type']])

        dataset[split]['s_id'].append(row['source_ID'])
        dataset[split]['t_id'].append(row['target_ID'])

        if dataset_name == 'cdcp_ACL17':
            i = 1
        else:
            i = 2
        s_index = int(row['source_ID'].split('_')[i])
        t_index = int(row['target_ID'].split('_')[i])
        difference = t_index - s_index

        difference_array = [0] * distance * 2
        if difference > distance:
            difference_array[-distance:] = [1] * distance
        elif difference < -distance:
            difference_array[:distance] = [1] * distance
        elif difference > 0:
            difference_array[-distance: distance + difference] = [1] * difference
        elif difference < 0:
            difference_array[distance + difference: distance] = [1] * -difference
        dataset[split]['distance'].append(difference_array)

        # load the document as list of argumentative component
        embed_length = 0
        text_embeddings = []
        text_mark = []
        for prop_id in range(0, 50):
            complete_prop_id = str(text_ID) + "_" + str(prop_id)
            file_path = os.path.join(embed_path, complete_prop_id + '.npz')

            if os.path.exists(file_path):
                embeddings = np.load(file_path)['arr_0']
                text_embeddings.extend(embeddings)

                prop_length = len(embeddings)

                # create the marks
                if complete_prop_id == source_ID:
                    prop_mark = [[1, 0]] * prop_length
                elif complete_prop_id == target_ID:
                    prop_mark = [[0, 1]] * prop_length
                else:
                    prop_mark = [[0, 0]] * prop_length
                text_mark.append(prop_mark)

        text_mark = np.concatenate(text_mark)
        dataset[split]['mark'].append(text_mark)
        # embeddings = np.concatenate([text_embeddings])
        dataset[split]['texts'].append(text_embeddings)
        embed_length = len(text_embeddings)
        if embed_length > max_text_len:
            max_text_len = embed_length

        """
        if dataset_name=='cdcp_ACL17':
            file_path = os.path.join(embed_path, "%05d" % (text_ID) + '.npz')
        else:
            file_path = os.path.join(embed_path, str(text_ID) + '.npz')
        embeddings = np.load(file_path)['arr_0']
        embed_length = len(embeddings)
        if embed_length > max_text_len:
            max_text_len = embed_length
        dataset[split]['texts'].append(embeddings)
        """

        file_path = os.path.join(embed_path, source_ID + '.npz')
        embeddings = np.load(file_path)['arr_0']
        embed_length = len(embeddings)
        if embed_length > max_prop_len:
            max_prop_len = embed_length
        dataset[split]['source_props'].append(embeddings)

        file_path = os.path.join(embed_path, target_ID + '.npz')
        embeddings = np.load(file_path)['arr_0']
        embed_length = len(embeddings)
        if embed_length > max_prop_len:
            max_prop_len = embed_length
        dataset[split]['target_props'].append(embeddings)

        if DEBUG and len(dataset['validation']['target_props'])>10:
            break

    print(str(time.ctime()) + '\t\tPADDING...')

    if feature_type == 'bow':
        pad = 0
        dtype = np.uint16
        ndim = 2
    elif feature_type == 'embeddings':
        pad = np.zeros(DIM)
        dtype = np.float32
        ndim = 3

    for split in ('train', 'validation', 'test'):

        dataset[split]['distance'] = np.array(dataset[split]['distance'], dtype=np.int8)


        print(str(time.ctime()) + '\t\t\tPADDING ' + split)

        texts = dataset[split]['texts']
        marks = dataset[split]['mark']
        for j in range(len(texts)):
            text = texts[j]
            mark = marks[j]
            embeddings = []
            new_marks = []
            diff = max_text_len - len(text)
            for i in range(diff):
                embeddings.append(pad)
                new_marks.append([0, 0] * 1)
            for embedding in text:
                embeddings.append(embedding)
            for old_mark in mark:
                new_marks.append(old_mark)
            texts[j] = embeddings
            marks[j] = new_marks

        dataset[split]['texts'] = np.array(texts, ndmin=ndim, dtype=dtype)
        dataset[split]['mark'] = np.array(marks, dtype=np.int8, ndmin=3)

        texts = dataset[split]['source_props']
        for j in range(len(texts)):
            text = texts[j]
            embeddings = []
            diff = max_prop_len - len(text)
            for i in range(diff):
                embeddings.append(pad)
            for embedding in text:
                embeddings.append(embedding)
            texts[j] = embeddings
        dataset[split]['source_props'] = np.array(texts, ndmin=ndim, dtype=dtype)

        texts = dataset[split]['target_props']
        for j in range(len(texts)):
            text = texts[j]
            embeddings = []
            diff = max_prop_len - len(text)
            for i in range(diff):
                embeddings.append(pad)
            for embedding in text:
                embeddings.append(embedding)
            texts[j] = embeddings
        dataset[split]['target_props'] = np.array(texts, ndmin=ndim, dtype=dtype)


    return dataset, max_text_len, max_prop_len


def perform_training(name = 'prova999',
                     save_weights_only=False,
                    use_conv = False,
                    epochs = 1000,
                    feature_type = 'bow',
                    patience = 200,
                    loss_weights = [20, 20, 1, 1],
                    lr_alfa = 0.003,
                    lr_kappa = 0.001,
                    beta_1 = 0.9,
                    beta_2 = 0.999,
                    res_size = 30,
                    resnet_layers = (2, 2),
                    embedding_size = 20,
                    embedder_layers = 2,
                    avg_pad = 20,
                    final_size = 20,
                    batch_size = 200,
                    regularizer_weight = 0.001,
                    dropout_resnet = 0.5,
                    dropout_embedder = 0.5,
                    dropout_final = 0,
                    cross_embed = False,
                    single_LSTM=False,
                    pooling=10,
                    text_pooling=10,
                    pooling_type='avg',
                    monitor="relations",
                    dataset_name = 'AAEC_v2',
                    dataset_version = 'new_2',
                    bn_embed=True,
                    bn_res=True,
                    bn_final=True,
                    network=2,
                     true_validation=False,
                     same_layers=False,
                     context=True,
                     distance=5,
                     temporalBN=False):

    if DEBUG:
        epochs = 20


    outputs_units = ()
    if dataset_name == 'AAEC_v2':
        outputs_units = (2, 5, 3, 3)
        min_text = 168
        min_prop = 72
    else:
        outputs_units = (2, 5, 5, 5)
        min_text = 552
        min_prop = 153

    print(str(time.ctime()) + "\tLAUNCHING TRAINING: " + name)
    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = load_dataset(dataset_name=dataset_name,
                                                       dataset_version=dataset_version,
                                                       dataset_split='total',
                                                       feature_type=feature_type,
                                                       min_text_len=min_text,
                                                       min_prop_len=min_prop,
                                                       distance=distance)
    print(str(time.ctime()) + "\tDATASET LOADED...")

    if distance > 0:
        distance = True

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
    X3_train = [X_text_train, X_source_train, X_target_train, X_dist_train, X_marks_train,]

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
    X3_test = [X_text_test, X_source_test, X_target_test, X_dist_test, X_marks_test,]

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
    X3_validation = [X_text_validation, X_source_validation, X_target_validation, X_dist_validation, X_marks_validation]

    print(str(time.ctime()) + "\t\tVALIDATION DATA PROCESSED...")
    print(str(time.ctime()) + "\t\tCREATING MODEL...")

    bow = None
    if feature_type == 'bow':
        dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
        vocabulary_path = os.path.join(dataset_path, 'glove', dataset_version,'glove.embeddings.npz')
        vocabulary_list = np.load(vocabulary_path)
        embed_list = vocabulary_list['embeds']
        word_list = vocabulary_list['vocab']

        bow = np.zeros((len(word_list) + 1, DIM))
        for index in range(len(word_list)):
            bow[index + 1] = embed_list[index]
        print(str(time.ctime()) + "\t\t\tEMBEDDINGS LOADED...")


    """
    model = build_net_1(bow=bow, text_length=max_text_len, propos_length=max_prop_len,
                        regularizer_weight=regularizer_weight,
                        resnet_layers=resnet_layers,
                        dropout_embedder=dropout_embedder,
                        dropout_resnet=dropout_resnet,
                        res_size=res_size,
                        embedding_size=embedding_size,
                        outputs=outputs_units,
                        use_conv=use_conv)
    """

    if network == 2:
        model = build_net_2(bow=bow,
                            cross_embed=cross_embed,
                            text_length=max_text_len, propos_length=max_prop_len,
                            regularizer_weight=regularizer_weight,
                            dropout_embedder=dropout_embedder,
                            dropout_resnet=dropout_resnet,
                            embedding_size=embedding_size,
                            embedder_layers=embedder_layers,
                            avg_pad=avg_pad,
                            resnet_layers=resnet_layers,
                            res_size=res_size,
                            final_size=final_size,
                            outputs=outputs_units,
                            bn_embed=bn_embed,
                            bn_res=bn_res,
                            bn_final=bn_final,
                            single_LSTM=single_LSTM,
                            pooling=pooling)
    elif network == 4:
        model = build_net_4(bow=bow,
                            text_length=max_text_len, propos_length=max_prop_len,
                            regularizer_weight=regularizer_weight,
                            dropout_embedder=dropout_embedder,
                            dropout_resnet=dropout_resnet,
                            embedding_size=embedding_size,
                            embedder_layers=embedder_layers,
                            resnet_layers=resnet_layers,
                            res_size=res_size,
                            final_size=final_size,
                            outputs=outputs_units,
                            bn_embed=bn_embed,
                            bn_res=bn_res,
                            bn_final=bn_final,
                            single_LSTM=single_LSTM,
                            pooling=pooling,
                            text_pooling=text_pooling,
                            pooling_type=pooling_type,
                            dropout_final=dropout_final)
    elif network == 5:
        model = build_net_5(bow=bow,
                            text_length=max_text_len, propos_length=max_prop_len,
                            regularizer_weight=regularizer_weight,
                            dropout_embedder=dropout_embedder,
                            dropout_resnet=dropout_resnet,
                            embedding_size=embedding_size,
                            embedder_layers=embedder_layers,
                            resnet_layers=resnet_layers,
                            res_size=res_size,
                            final_size=final_size,
                            outputs=outputs_units,
                            bn_embed=bn_embed,
                            bn_res=bn_res,
                            bn_final=bn_final,
                            single_LSTM=single_LSTM,
                            pooling=pooling,
                            text_pooling=text_pooling,
                            pooling_type=pooling_type,
                            dropout_final=dropout_final)
    elif network == 7:
        model = build_net_7(bow=bow,
                            text_length=max_text_len, propos_length=max_prop_len,
                            regularizer_weight=regularizer_weight,
                            dropout_embedder=dropout_embedder,
                            dropout_resnet=dropout_resnet,
                            embedding_size=embedding_size,
                            embedder_layers=embedder_layers,
                            resnet_layers=resnet_layers,
                            res_size=res_size,
                            final_size=final_size,
                            outputs=outputs_units,
                            bn_embed=bn_embed,
                            bn_res=bn_res,
                            bn_final=bn_final,
                            single_LSTM=single_LSTM,
                            pooling=pooling,
                            text_pooling=text_pooling,
                            pooling_type=pooling_type,
                            dropout_final=dropout_final,
                            same_DE_layers=same_layers,
                            context=context,
                            distance=distance,
                            temporalBN=temporalBN)

    # plot_model(model, to_file='model.png', show_shapes=True)

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
    #elif dataset_name == 'AAEC_v2':
    #    props_fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_0_1_2]
    elif dataset_name == 'AAEC_v2':
        props_fmeasures = [fmeasure_0_1_2]

    # for using them during model loading
    custom_objects = {}
    for fmeasure in fmeasures:
        custom_objects[fmeasure.__name__] = fmeasure

    crop0 = create_crop_fn(1, 0, 1)
    crop1 = create_crop_fn(1, 1, 2)
    crop2 = create_crop_fn(1, 2, 3)
    crop3 = create_crop_fn(1, 3, 4)
    crop4 = create_crop_fn(1, 4, 5)

    crops = [crop0, crop1, crop2, crop3, crop4]
    for crop in crops:
        custom_objects[crop.__name__] = crop

    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    lr_function = create_lr_annealing_function(initial_lr=lr_alfa, k=lr_kappa)

    model.compile(loss='categorical_crossentropy',
                  loss_weights=loss_weights,
                  optimizer=Adam(lr=lr_function(0),
                                 beta_1=beta_1,
                                 beta_2=beta_2),
                  metrics={'link': [fmeasure_0],
                           # 'relation': [fmeasure_0, fmeasure_2, fmeasure_0_2, fmeasure_0_1_2_3],
                           'relation': [fmeasure_0_2, fmeasure_0_1_2_3],
                           'source': props_fmeasures,
                           'target': props_fmeasures}
                  )

    model.summary()

    # PERSISTENCE CONFIGURATION
    if not save_weights_only:
        complete_network_name = name + '_completemodel.{epoch:03d}.h5'
    else:
        model_name = name + '_model.json'
        json_model = model.to_json()
        with open(os.path.join(save_dir, model_name), 'w') as outfile:
            json.dump(json_model, outfile)
        weights_name = name + '_weights.{epoch:03d}.h5'

    if not save_weights_only:
        file_path = os.path.join(save_dir, complete_network_name)
    else:
        file_path = os.path.join(save_dir, weights_name)
    
    if monitor == 'relations':
        monitor='val_relation_' + fmeasure_0_1_2_3.__name__
    elif monitor == 'pos_relations':
        monitor='val_relation_' + fmeasure_0_2.__name__
    elif monitor == 'links':
        monitor='val_link_' + fmeasure_0.__name__

    log_path = os.path.join(save_dir, name + '_training.log')
    logger = CSVLogger(log_path, separator='\t', append=False)
    timer = TimingCallback()

    if true_validation:
        # modify the lr each epoch
        logger = CSVLogger(log_path, separator='\t', append=True)

        best_score = 0
        waited = 0

        log_path = os.path.join(save_dir, name + '_validation.log')
        val_file = open(log_path,'w')

        """
        real_validation = RealValidationCallback(patience,
                                                 os.path.join(save_dir, name + '_validationCB.log'),
                                                 file_path + "b")
        """

        if dataset_name == "AAEC_v2":
            val_file.write("set\tAVG all\tAVG LP\tlink\tR AVG dir\tR support\tR attack\t" +
                           "P AVG\tP premise\tP claim\tP major claim\n")
        elif dataset_name == "cdcp_ACL17":
            val_file.write("set\tAVG all\tAVG LP\tlink\tR AVG dir\tR reason\tR evidence\t" +
                           "P AVG\tP policy\tP fact\tP testimony\tP value\tP reference\n")

        last_epoch = 0
        for epoch in range(1, epochs+1):
            print("\nEPOCH: " + str(epoch))
            lr_annealing_fn = create_lr_annealing_function(initial_lr=lr_alfa, k=lr_kappa, fixed_epoch=epoch)
            lr_scheduler = LearningRateScheduler(lr_annealing_fn)
            callbacks = [lr_scheduler, logger, timer]
            model.fit(x=X3_train,
                      # y=Y_links_train,
                      y=Y_train,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                      callbacks=callbacks,
                      initial_epoch=epoch
                      )

            # evaluation
            Y_pred = model.predict(X3_validation)
            Y_pred_prop = np.concatenate([Y_pred[2], Y_pred[3]])
            Y_test_prop = np.concatenate([Y_validation[2], Y_validation[3]])
            Y_pred_prop = np.argmax(Y_pred_prop, axis=-1)
            Y_test_prop = np.argmax(Y_test_prop, axis=-1)
            Y_pred_links = np.argmax(Y_pred[0], axis=-1)
            Y_test_links = np.argmax(Y_validation[0], axis=-1)
            Y_pred_rel = np.argmax(Y_pred[1], axis=-1)
            Y_test_rel = np.argmax(Y_validation[1], axis=-1)

            score_link = f1_score(Y_test_links, Y_pred_links, average=None, labels=[0])
            score_rel = f1_score(Y_test_rel, Y_pred_rel, average=None, labels=[0, 2])
            score_rel_AVG = f1_score(Y_test_rel, Y_pred_rel, average='macro', labels=[0, 2])
            score_prop = f1_score(Y_test_prop, Y_pred_prop, average=None)
            score_prop_AVG = f1_score(Y_test_prop, Y_pred_prop, average='macro')

            score_AVG_LP = np.mean([score_link, score_prop_AVG])
            score_AVG_all = np.mean([score_link, score_prop_AVG, score_rel_AVG])


            string = str(epoch) + "\t" + str(round(score_AVG_all[0], 5)) + "\t" + str(round(score_AVG_LP[0], 5))
            string += "\t" + str(round(score_link[0], 5)) + "\t" + str(round(score_rel_AVG, 5))
            for score in score_rel:
                string += "\t" + str(round(score, 5))
            string += "\t" + str(round(score_prop_AVG, 5))
            for score in score_prop:
                string += "\t" + str(round(score, 5))

            if score_link > best_score:
                best_score = score_link
                string += "\t!"

                file_path = os.path.join(save_dir, name + '_weights.%03d.h5' % epoch)
                # save
                print("Saving to " + file_path)
                model.save_weights(file_path)
                waited = 0
            else:
                waited += 1
                # early stopping
                if waited > patience:
                    break

            val_file.write(string + "\n")

            val_file.flush()
            last_epoch = epoch
        val_file.close()

    else:

        # save the networks each epoch
        checkpoint = ModelCheckpoint(filepath=file_path,
                                     # monitor='val_loss',
                                     monitor=monitor,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=save_weights_only,
                                     mode='max'
                                     )

        # modify the lr each epoch
        lr_scheduler = LearningRateScheduler(lr_function)

        # early stopping
        early_stop = EarlyStopping(patience=patience,
                                   # monitor='val_loss',
                                   monitor=monitor,
                                   verbose=2,
                                   mode='max')

        callbacks = [checkpoint, early_stop, lr_scheduler, logger, timer]

        print(str(time.ctime()) + "\tSTARTING TRAINING")

        sys.stdout.flush()

        history = model.fit(x=X3_train,
                            # y=Y_links_train,
                            y=Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            # validation_data=(X_validation, Y_links_validation),
                            validation_data=(X3_validation, Y_validation),
                            callbacks=callbacks
                            )

        print(str(time.ctime()) + "\tTRAINING FINISHED")

        last_epoch = len(history.epoch)
    print("\n-----------------------\n")

    print(str(time.ctime()) + "\tEVALUATING MODEL")

    last_path = ""
    

    if save_weights_only:
        model = model_from_json(json_model, custom_objects=custom_objects)

        for epoch in range(last_epoch, 0, -1):
            netpath = os.path.join(save_dir, name + '_weights.%03d.h5' % epoch)
            if os.path.exists(netpath):
                last_path = netpath
                break
        model.load_weights(last_path)

    else:
        for epoch in range(last_epoch, 0, -1):
            netpath = os.path.join(save_dir, name + '_completemodel.%03d.h5' % epoch)
            if os.path.exists(netpath):
                last_path = netpath
                break

        model = load_model(last_path, custom_objects=custom_objects)

    print("\n\n\tLOADED NETWORK: " + last_path + "\n")


    if context and distance:
        X = {'test': X3_test,
             'train': X3_train,
             'validation': X3_validation}
    elif distance:
        X = {'test': X3_test[1:-1],
             'train': X3_train[1:-1],
             'validation': X3_validation[1:-1]}
    elif context:
        X = {'test': X3_test[1:-2, -1:],
             'train': X3_train[1:-2, -1:],
             'validation': X3_validation[1:-2, -1:]}
    else:
        X = {'test': X3_test[1:-2],
             'train': X3_train[1:-2],
             'validation': X3_validation[1:-2]}

    Y = {'test': Y_test,
         'train': Y_train,
         'validation': Y_validation}

    testfile = open(os.path.join(save_dir, name + "_eval.txt"), 'w')

    print("\n-----------------------\n")

    if dataset_name=="AAEC_v2":
        testfile.write("set\tAVG all\tAVG LP\tlink\tR AVG dir\tR support\tR attack\t" +
                       "P AVG\tP premise\tP claim\tP major claim\n\n")
    elif dataset_name=="cdcp_ACL17":
        testfile.write("set\tAVG all\tAVG LP\tlink\tR AVG dir\tR reason\tR evidence\t" +
                       "P AVG\tP policy\tP fact\tP testimony\tP value\tP reference\n\n")
        


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
        string = split + "\t" + str(score_AVG_all_real[0]) + "\t" + str(score_AVG_LP_real[0])
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



if __name__ == '__main__':
    if DEBUG:
        print("DEBUG! DEBUG! DEBUG!!!!")
        print("DEBUG! DEBUG! DEBUG!!!!")
        print("DEBUG! DEBUG! DEBUG!!!!")
        print("DEBUG! DEBUG! DEBUG!!!!")
    name = 'prova999'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    dataset_name = 'cdcp_ACL17'
    dataset_version = 'new_3'
    split = 'test'

    perform_training(
        name='cdcp7R13tbn',
        save_weights_only=True,
        use_conv=False,
        epochs=1000,
        feature_type='bow',
        patience=200,
        loss_weights=[0, 10, 1, 1],
        lr_alfa=0.005,
        lr_kappa=0.001,
        beta_1=0.9,
        beta_2=0.9999,
        res_size=5,
        resnet_layers=(1, 2),
        embedding_size=50,
        embedder_layers=4,
        avg_pad=20,
        final_size=20,
        batch_size=500,
        regularizer_weight=0.0001,
        dropout_resnet=0.1,
        dropout_embedder=0.1,
        dropout_final=0.1,
        bn_embed=True,
        bn_res=True,
        bn_final=True,
        cross_embed=False,
        single_LSTM=True,
        pooling=10,
        text_pooling=50,
        network=7,
        monitor="links",
        true_validation=False,
        temporalBN = True,
        same_layers = False,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )

    name = 'cdcp7R13tbn'
    evaluate_net.evaluate_every_epoch(name, dataset_name, dataset_version)
    
    perform_training(
        name='cdcp7R13sde',
        save_weights_only=True,
        use_conv=False,
        epochs=1000,
        feature_type='bow',
        patience=200,
        loss_weights=[0, 10, 1, 1],
        lr_alfa=0.005,
        lr_kappa=0.001,
        beta_1=0.9,
        beta_2=0.9999,
        res_size=5,
        resnet_layers=(1, 2),
        embedding_size=50,
        embedder_layers=4,
        avg_pad=20,
        final_size=20,
        batch_size=500,
        regularizer_weight=0.0001,
        dropout_resnet=0.1,
        dropout_embedder=0.1,
        dropout_final=0.1,
        bn_embed=True,
        bn_res=True,
        bn_final=True,
        cross_embed=False,
        single_LSTM=True,
        pooling=10,
        text_pooling=50,
        network=7,
        monitor="links",
        true_validation=False,
        temporalBN = False,
        same_layers = True,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )

    name = 'cdcp7R13sde'
    evaluate_net.evaluate_every_epoch(name, dataset_name, dataset_version)
    
    perform_training(
        name='cdcp7R13tbnsde',
        save_weights_only=True,
        use_conv=False,
        epochs=1000,
        feature_type='bow',
        patience=200,
        loss_weights=[0, 10, 1, 1],
        lr_alfa=0.005,
        lr_kappa=0.001,
        beta_1=0.9,
        beta_2=0.9999,
        res_size=5,
        resnet_layers=(1, 2),
        embedding_size=50,
        embedder_layers=4,
        avg_pad=20,
        final_size=20,
        batch_size=500,
        regularizer_weight=0.0001,
        dropout_resnet=0.1,
        dropout_embedder=0.1,
        dropout_final=0.1,
        bn_embed=True,
        bn_res=True,
        bn_final=True,
        cross_embed=False,
        single_LSTM=True,
        pooling=10,
        text_pooling=50,
        network=7,
        monitor="links",
        true_validation=False,
        temporalBN = True,
        same_layers = True,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )

    name = 'cdcp7R13tbnsde'
    evaluate_net.evaluate_every_epoch(name, dataset_name, dataset_version)
    
    perform_training(
        name='cdcp7R13all',
        save_weights_only=True,
        use_conv=False,
        epochs=1000,
        feature_type='bow',
        patience=200,
        loss_weights=[0, 10, 1, 1],
        lr_alfa=0.005,
        lr_kappa=0.001,
        beta_1=0.9,
        beta_2=0.9999,
        res_size=5,
        resnet_layers=(1, 2),
        embedding_size=50,
        embedder_layers=4,
        avg_pad=20,
        final_size=20,
        batch_size=500,
        regularizer_weight=0.0001,
        dropout_resnet=0.1,
        dropout_embedder=0.1,
        dropout_final=0.1,
        bn_embed=True,
        bn_res=True,
        bn_final=True,
        cross_embed=False,
        single_LSTM=True,
        pooling=10,
        text_pooling=50,
        network=7,
        monitor="links",
        true_validation=True,
        temporalBN = True,
        same_layers = True,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )

    name = 'cdcp7R13all'
    evaluate_net.evaluate_every_epoch(name, dataset_name, dataset_version)
    
    # more weight
    perform_training(
        name='cdcp7R16',
        save_weights_only=True,
        use_conv=False,
        epochs=1000,
        feature_type='bow',
        patience=200,
        loss_weights=[0, 10, 1, 1],
        lr_alfa=0.005,
        lr_kappa=0.001,
        beta_1=0.9,
        beta_2=0.9999,
        res_size=5,
        resnet_layers=(1, 2),
        embedding_size=50,
        embedder_layers=4,
        avg_pad=20,
        final_size=20,
        batch_size=500,
        regularizer_weight=0.1,
        dropout_resnet=0.1,
        dropout_embedder=0.1,
        dropout_final=0.1,
        bn_embed=True,
        bn_res=True,
        bn_final=True,
        cross_embed=False,
        single_LSTM=True,
        pooling=10,
        text_pooling=50,
        network=7,
        monitor="links",
        true_validation=True,
        temporalBN = True,
        same_layers = True,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )

    name = 'cdcp7R16'
    evaluate_net.evaluate_every_epoch(name, dataset_name, dataset_version)
    
    # more! weight
    perform_training(
        name='cdcp7R17',
        save_weights_only=True,
        use_conv=False,
        epochs=1000,
        feature_type='bow',
        patience=200,
        loss_weights=[0, 10, 1, 1],
        lr_alfa=0.005,
        lr_kappa=0.001,
        beta_1=0.9,
        beta_2=0.9999,
        res_size=5,
        resnet_layers=(1, 2),
        embedding_size=50,
        embedder_layers=4,
        avg_pad=20,
        final_size=20,
        batch_size=500,
        regularizer_weight=10,
        dropout_resnet=0.1,
        dropout_embedder=0.1,
        dropout_final=0.1,
        bn_embed=True,
        bn_res=True,
        bn_final=True,
        cross_embed=False,
        single_LSTM=True,
        pooling=10,
        text_pooling=50,
        network=7,
        monitor="links",
        true_validation=True,
        temporalBN = True,
        same_layers = True,
        dataset_name=dataset_name,
        dataset_version=dataset_version
    )

    name = 'cdcp7R17'
    evaluate_net.evaluate_every_epoch(name, dataset_name, dataset_version)
    
    
    
    