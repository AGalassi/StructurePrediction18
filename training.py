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

from networks import build_net_1, build_net_2
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.preprocessing.sequence import  pad_sequences
from training_utils import TimingCallback, lr_annealing, fmeasure, get_avgF1
from glove_loader import DIM
from sklearn.metrics import f1_score

DIST_MAX = 5

def load_dataset(dataset_split='total', dataset_name='cdcp_ACL17', dataset_version='new_2',
                 feature_type='embeddings'):

    max_prop_len = 0
    max_text_len = 0

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

    for index, row in df.iterrows():

        text_ID = row['text_ID']
        source_ID = row['source_ID']
        target_ID = row['target_ID']
        split = row['set']

        dataset[split]['sources_type'].append(categorical_prop[row['source_type']])
        dataset[split]['targets_type'].append(categorical_prop[row['target_type']])
        dataset[split]['relations_type'].append(categorical_link[row['relation_type']])

        if row['source_to_target']:
            dataset[split]['links'].append([1, 0])
        else:
            dataset[split]['links'].append([0, 1])

        if dataset_name == 'cdcp_ACL17':
            i = 1
        else:
            i = 2
        s_index = int(row['source_ID'].split('_')[i])
        t_index = int(row['target_ID'].split('_')[i])
        difference = t_index - s_index

        difference_array = [0] * DIST_MAX * 2
        if difference > DIST_MAX:
            difference_array[-DIST_MAX:] = [1] * DIST_MAX
        elif difference < -DIST_MAX:
            difference_array[:DIST_MAX] = [1] * DIST_MAX
        elif difference > 0:
            difference_array[-DIST_MAX: DIST_MAX + difference] = [1] * difference
        elif difference < 0:
            difference_array[DIST_MAX + difference: DIST_MAX] = [1] * -difference
        dataset[split]['distance'].append(difference_array)

        # load the document as list of argumentative component
        embed_length = 0
        text_embeddings = []
        text_mark = []
        for prop_id in range(0, 50):
            complete_prop_id = text_ID + "_" + str(prop_id)
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

        if split == 'validation':
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


if __name__ == '__main__':

    name = 'prova999'
    if len(sys.argv) > 1:
        name = sys.argv[1]
    print(str(time.ctime()) + "\tLAUNCHING TRAINING: " + name)

    batch_size = 500
    epochs = 1
    patience = 200
    save_weights_only = False
    lr_alfa = 0.003
    lr_kappa = 0.001
    beta_1 = 0.9
    beta_2 = 0.999  
    regularizer_weight = 0.01
    loss_weights = [20, 20, 1, 1]
    feature_type = 'bow'
    res_size = 30
    resnet_layers = (2, 2)
    embedding_size = 20
    use_conv = False
    dropout_resnet = 0.3
    dropout_embedder = 0.1
    embedder_layers = 2
    avg_pad = 20
    final_size = 20

    # dataset_name = 'cdcp_ACL17'
    # dataset_version = 'new_3'
    
    dataset_name = 'AAEC_v2'
    dataset_version = 'new_2'
    

    outputs_units = ()
    if dataset_name == 'AAEC_v2':
        outputs_units = (2, 5, 3, 3)
    else:
        outputs_units = (2, 5, 5, 5)

    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = load_dataset(dataset_name=dataset_name,
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

    dataset = 0
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

    model = build_net_2(bow=bow,
                        cross_embed=True,
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
                        outputs=outputs_units)

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
        props_fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_3, fmeasure_4, fmeasure_0_1_2_3_4]
    elif dataset_name == 'AAEC_v2':
        props_fmeasures = [fmeasure_0, fmeasure_1, fmeasure_2, fmeasure_0_1_2]

    fmeasures_names = {}
    for fmeasure in fmeasures:
        fmeasures_names[fmeasure.__name__] = fmeasure

    model.compile(loss='categorical_crossentropy',
                  loss_weights=loss_weights,
                  optimizer=Adam(lr=lr_annealing(0, lr_alfa, lr_kappa),
                                 beta_1=beta_1,
                                 beta_2=beta_2),
                  metrics={'link': [fmeasure_0],
                           'relation': [fmeasure_0, fmeasure_2, fmeasure_0_2, fmeasure_0_1_2_3],
                           'source': props_fmeasures,
                           'target': props_fmeasures}
                  )

    model.summary()

    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)

    complete_network_name = name + '_completemodel.{epoch:03d}.h5'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, complete_network_name)

    # save the networks each epoch
    checkpoint = ModelCheckpoint(filepath=file_path,
                                 # monitor='val_loss',
                                 monitor='val_relation_' + fmeasure_0_1_2_3.__name__,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='max'
                                 )

    # modify the lr each epoch
    lr_scheduler = LearningRateScheduler(lr_annealing)

    # early stopping
    early_stop = EarlyStopping(patience=patience,
                               # monitor='val_loss',
                               monitor='val_relation_' + fmeasure_0_1_2_3.__name__,
                               verbose=2,
                               mode='max')

    logger = CSVLogger(os.path.join(save_dir, name + '_training.log'), separator='\t', append=False)

    timer = TimingCallback()

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

    print(str(time.ctime()) + "\tEVALUATING MODEL")

    last_path = ""
    for epoch in range(len(history.epoch), 0, -1):
        netpath = os.path.join(save_dir, name + '_completemodel.%03d.h5' % epoch)
        if os.path.exists(netpath):
            last_path = netpath
            break

    model = load_model(last_path,
                       custom_objects=fmeasures_names
                       )

    score = model.evaluate(X3_test, Y_test, verbose=0)

    string = ""
    for metric in model.metrics_names:
        string += metric + "\t"
    print(string)

    string = ""
    for metric in score:
        string += str(metric) + "\t"
    print(string)

    print(str(time.ctime()) + "\t\tLABEL PREDICTION")

    Y_pred = model.predict(X3_test)

    print(Y_pred)

    Y_pred_tot = np.concatenate([Y_pred[2], Y_pred[3]])
    Y_test_tot = np.concatenate([Y_stype_test, Y_ttype_test])

    Y_pred_labels = []
    Y_labels_test = []

    for scores in Y_pred_tot:
        Y_pred_labels.append(np.argmax(scores))
    for scores in Y_test_tot:
        Y_labels_test.append(np.argmax(scores))

    print("\n")
    print(Y_pred_labels)

    func_scores = f1_score(Y_labels_test, Y_pred_labels, average=None)
    string = ""
    for i in range(len(func_scores)):
        string += str(i) + "\t"
    string += "AVG\n"

    for metric in func_scores:
        string += str(metric) + "\t"

    string += f1_score(Y_labels_test, Y_pred_labels)
    print(string)


