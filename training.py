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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
from keras.preprocessing.sequence import  pad_sequences
from training_utils import TimingCallback, lr_annealing

DIM = 300


def load_dataset(dataset_split='total', dataset_name='cdcp_ACL17', dataset_version='new_2'):

    max_prop_len = 0
    max_text_len = 0

    dataset_path = os.path.join(os.getcwd(), 'Datasets', dataset_name)
    dataframe_path = os.path.join(dataset_path, 'pickles', dataset_version, dataset_split + '.pkl')
    embed_path = os.path.join(dataset_path, 'embeddings', dataset_version)

    df = pandas.read_pickle(dataframe_path)

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

        file_path = os.path.join(embed_path, "%05d" % (text_ID) + '.npz')
        embeddings = np.load(file_path)['arr_0']
        embed_length = len(embeddings)
        if embed_length > max_text_len:
            max_text_len = embed_length
        dataset[split]['texts'].append(embeddings)

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

    print(str(time.ctime()) + '\t\tPADDING...')
    for split in ('train', 'validation', 'test'):

        print(str(time.ctime()) + '\t\t\tPADDING ' + split)

        texts = dataset[split]['texts']
        for j in range(len(texts)):
            text = texts[j]
            embeddings = []
            diff = max_text_len - len(text)
            for i in range(diff):
                embeddings.append(np.zeros(DIM, dtype=np.float32))
            for embedding in text:
                embeddings.append(embedding)
            texts[j] = embeddings

        dataset[split]['texts'] = np.array(dataset[split]['texts'], ndmin=3, dtype=np.float32)

        texts = dataset[split]['source_props']
        for j in range(len(texts)):
            text = texts[j]
            embeddings = []
            diff = max_prop_len - len(text)
            for i in range(diff):
                embeddings.append(np.zeros(DIM))
            for embedding in text:
                embeddings.append(embedding)
            texts[j] = embeddings
        dataset[split]['source_props'] = np.array(dataset[split]['source_props'], ndmin=3, dtype=np.float32)

        texts = dataset[split]['target_props']
        for j in range(len(texts)):
            text = texts[j]
            embeddings = []
            diff = max_prop_len - len(text)
            for i in range(diff):
                embeddings.append(np.zeros(DIM))
            for embedding in text:
                embeddings.append(embedding)
            texts[j] = embeddings
        dataset[split]['target_props'] = np.array(dataset[split]['target_props'], ndmin=3, dtype=np.float32)

    text = 0
    texts = 0

    return dataset, max_text_len, max_prop_len


if __name__ == '__main__':

    name = 'prova'
    if len(sys.argv) > 1:
        name = sys.argv[1]
    print(str(time.ctime()) + "\tLAUNCHING TRAINING: " + name)

    batch_size = 500
    epochs = 1000
    patience = 200
    save_weights_only = False
    lr_alfa = 0.001
    lr_kappa = 0.001
    beta_1 = 0.9
    beta_2 = 0.999


    dataset_name = 'cdcp_ACL17'
    dataset_version = 'new_2'

    print(str(time.ctime()) + "\tLOADING DATASET...")
    dataset, max_text_len, max_prop_len = load_dataset(dataset_name=dataset_name, dataset_version=dataset_version)
    print(str(time.ctime()) + "\tDATASET LOADED...")

    sys.stdout.flush()

    print(str(time.ctime()) + "\tPROCESSING DATA AND MODEL...")

    split = 'train'
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

    print(str(time.ctime()) + "\t\tVALIDATION DATA PROCESSED...")

    dataset = 0
    model = build_net_1(text_length=max_text_len, propos_length=max_prop_len)
    # plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=lr_annealing(0, lr_alfa, lr_kappa),
                                 beta_1=beta_1,
                                 beta_2=beta_2),
                  metrics=['accuracy'])

    save_dir = os.path.join(os.getcwd(), 'network_models', dataset_name, dataset_version, name)

    complete_network_name = name + '_completemodel.{epoch:03d}.h5'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, complete_network_name)

    # save the networks each epoch
    checkpoint = ModelCheckpoint(filepath=file_path,
                                 monitor='val_link_output_L_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False
                                 )

    # modify the lr each epoch
    lr_scheduler = LearningRateScheduler(lr_annealing)

    # early stopping
    early_stop = EarlyStopping(monitor='val_link_output_L_acc', patience=patience, verbose=1)

    logger = CSVLogger(os.path.join(save_dir, name + '_training.log'), separator='\t', append=False)

    timer = TimingCallback()

    callbacks = [checkpoint, early_stop, lr_scheduler, logger, timer]

    print(str(time.ctime()) + "\tSTARTING TRAINING")

    history = model.fit(x=X_train,
                        # y=Y_links_train,
                        y=Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        # validation_data=(X_validation, Y_links_validation),
                        validation_data=(X_validation, Y_validation),
                        callbacks=callbacks
                        )

    print(str(time.ctime()) + "\tTRAINING FINISHED")
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(model.metrics_names)
    print(score)




