__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import keras
import numpy as np
from keras.layers import (BatchNormalization, Dropout, Dense, Input, Activation, LSTM, Conv1D, Add, Lambda, MaxPool1D,
                          Bidirectional, Concatenate, Flatten, Embedding, TimeDistributed, AveragePooling1D,
                          GlobalAveragePooling1D, GlobalMaxPooling1D)
from keras.utils.vis_utils import plot_model
import pydot
from glove_loader import DIM


def build_net_1(bow=None,
                text_length=200, propos_length=100,
                regularizer_weight=0.001,
                use_conv=True,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                embedding_size=int(DIM/3),
                embedder_layers=2,
                res_size=int(DIM/3),
                resnet_layers=(3, 2),
                final_size=int(DIM/3),
                outputs=(2, 5, 5, 5)):

    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="sourceprop_input_L")
        targetprop_il = Input(shape=(propos_length,), name="targetprop_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="sourceprop_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="targetprop_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="sourceprop_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="targetprop_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    text_LSTM = make_embedder(prev_text_l, 'text', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder, use_conv=use_conv)
    sourceprop_LSTM = make_embedder(prev_source_l, 'sourceprop', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder, use_conv=use_conv)
    targetprop_LSTM = make_embedder(prev_target_l, 'targetprop', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder, use_conv=use_conv)

    concat_l = Concatenate(name='embed_merge')([text_LSTM, sourceprop_LSTM, targetprop_LSTM])

    prev_l = make_resnet(concat_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet)

    prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='final_dropout')(prev_l)

    link_ol = Dense(units=outputs[0],
                    name='link',
                    activation='softmax',
                    )(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def build_net_3(bow=None,
                text_length=200, propos_length=100,
                regularizer_weight=0.001,
                use_conv=True,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                embedding_size=int(DIM/3),
                embedder_layers=2,
                res_size=int(DIM/3),
                resnet_layers=(3, 2),
                final_size=int(DIM/3),
                outputs=(2, 5, 5, 5)):

    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="sourceprop_input_L")
        targetprop_il = Input(shape=(propos_length,), name="targetprop_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="sourceprop_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="targetprop_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="sourceprop_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="targetprop_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    mark_il = Input(shape=(text_length, 2), name="mark_input_L")
    dist_il = Input(shape=(10,), name="dist_input_L")

    prev_text_l = Concatenate(name="mark_concatenation")([prev_text_l, mark_il])

    text_LSTM = make_embedder(prev_text_l, 'text', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder, use_conv=use_conv)
    sourceprop_LSTM = make_embedder(prev_source_l, 'sourceprop', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder, use_conv=use_conv)
    targetprop_LSTM = make_embedder(prev_target_l, 'targetprop', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder, use_conv=use_conv)

    concat_l = Concatenate(name='embed_merge')([text_LSTM, sourceprop_LSTM, targetprop_LSTM, dist_il])

    prev_l = make_resnet(concat_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet)

    prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='final_dropout')(prev_l)

    link_ol = Dense(units=outputs[0],
                    name='link',
                    activation='softmax',
                    )(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il, mark_il, dist_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def make_resnet(input_layer, regularizer_weight, layers=(2, 2), res_size=int(DIM/3)*3, dropout=0, bn=True):
    prev_layer = input_layer
    prev_block = prev_layer
    blocks = layers[0]
    res_layers = layers[1]

    shape = int(np.shape(input_layer)[1])

    for i in range(1, blocks + 1):
        for j in range(1, res_layers):
            if bn:
                prev_layer = BatchNormalization(name='resent_BN_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dropout(dropout, name='resnet_Dropout_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dense(units=res_size,
                               activation=None,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                               bias_regularizer=keras.regularizers.l2(regularizer_weight),
                               name='resnet_dense_' + str(i) + '_' + str(j)
                               )(prev_layer)
        if bn:
            prev_layer = BatchNormalization(name='BN_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dropout(dropout, name='resnet_Dropout_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dense(units=shape,
                           activation=None,
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                           bias_regularizer=keras.regularizers.l2(regularizer_weight),
                           name='resnet_dense_' + str(i) + '_' + str(res_layers)
                           )(prev_layer)

        prev_layer = Add(name='resnet_sum' + str(i))([prev_block, prev_layer])
        prev_block = prev_layer

    return prev_block


def make_embedder(input_layer, layer_name, regularizer_weight,
                  layers=2, layers_size=int(DIM/10), embedding_size=int(DIM/3), dropout=0, use_conv=True):
    prev_layer = input_layer

    shape = int(np.shape(input_layer)[2])
    for i in range(1, layers):

        prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(i))(prev_layer)

        prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(i))(prev_layer)

        prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(i))(prev_layer)

        if use_conv:

            prev_layer = Conv1D(filters=layers_size,
                                kernel_size=3,
                                padding='same',
                                activation=None,
                                kernel_initializer='he_normal',
                                kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                name=layer_name + '_conv_' + str(i)
                                )(prev_layer)
        else:
            prev_layer = TimeDistributed(Dense(units=layers_size,
                                               activation=None,
                                               kernel_initializer='he_normal',
                                               kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                               bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                               name=layer_name + '_dense_' + str(i)
                                               ))(prev_layer)

    prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(layers))(prev_layer)

    prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(layers))(prev_layer)

    prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(layers))(prev_layer)

    if use_conv:
        prev_layer = Conv1D(filters=shape,
                            kernel_size=3,
                            padding='same',
                            activation=None,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                            bias_regularizer=keras.regularizers.l2(regularizer_weight),
                            name=layer_name + '_conv_' + str(layers)
                            )(prev_layer)
    else:
        prev_layer = TimeDistributed(Dense(units=shape,
                                           activation=None,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                           bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                           name=layer_name + '_dense_' + str(layers)
                                           ))(prev_layer)

    prev_layer = Add(name=layer_name + '_sum')([input_layer, prev_layer])

    prev_layer = BatchNormalization(name=layer_name + '_BN')(prev_layer)

    text_LSTM = Bidirectional(LSTM(units=embedding_size,
                                   dropout=dropout,
                                   recurrent_dropout=dropout,
                                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                   recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                   return_sequences=False,
                                   unroll=False, # not possible to unroll if the time shape is not specified
                                   name=layer_name + '_LSTM'),
                              merge_mode='mul',
                              )(prev_layer)

    return text_LSTM


def make_deep_word_embedder(input_layer, layer_name, regularizer_weight,
                            layers=2, layers_size=int(DIM/10), dropout=0, bn=True):
    prev_layer = input_layer

    shape = int(np.shape(input_layer)[2])
    for i in range(1, layers):

        if bn:
            prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(i))(prev_layer)

        prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(i))(prev_layer)

        prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(i))(prev_layer)

        prev_layer = TimeDistributed(Dense(units=layers_size,
                                           activation=None,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                           bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                           name=layer_name + '_dense_' + str(i)
                                           ),
                                     name=layer_name + '_TD_' + str(i))(prev_layer)
    if bn:
        prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(layers))(prev_layer)

    prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(layers))(prev_layer)

    prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(layers))(prev_layer)

    prev_layer = TimeDistributed(Dense(units=shape,
                                       activation=None,
                                       kernel_initializer='he_normal',
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       name=layer_name + '_dense_' + str(layers)
                                       ),
                                 name=layer_name + '_TD_' + str(layers))(prev_layer)

    prev_layer = Add(name=layer_name + '_sum')([input_layer, prev_layer])

    return prev_layer


def make_embedder_layers(regularizer_weight, shape, layers=2, layers_size=int(DIM/10), dropout=0.1,
                         temporalBN=False):
    bn_list_prop = []
    layers_list = []
    dropout_list = []
    activation_list = []
    bn_list_text = []

    if layers > 0:
        layer = Dense(units=shape,
                      activation=None,
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                      bias_regularizer=keras.regularizers.l2(regularizer_weight),
                      name='dense_' + str(layers))
        layers_list.append(layer)
        if temporalBN:
            bn_list_prop.append(BatchNormalization(axis=-2, name="TBN_prop_" + str(layers)))
            bn_list_text.append(BatchNormalization(axis=-2, name="TBN_text_" + str(layers)))
        else:
            bn_list_prop.append(BatchNormalization(name="BN_" + str(layers)))
        dropout_list.append(Dropout(dropout, name='Dropout_' + str(layers)))
        activation_list.append(Activation('relu', name='ReLU_' + str(layers)))

    for i in range(1, layers):
        layer = Dense(units=layers_size,
                      activation=None,
                      kernel_initializer='he_normal',
                      kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                      bias_regularizer=keras.regularizers.l2(regularizer_weight),
                      name='dense_' + str(i))
        if temporalBN:
            bn_list_prop.append(BatchNormalization(axis=-2, name="TBN_prop_" + str(i)))
            bn_list_text.append(BatchNormalization(axis=-2, name="TBN_text_" + str(i)))
        else:
            bn_list_prop.append(BatchNormalization(name="BN_" + str(i)))
        layers_list.append(layer)
        dropout_list.append(Dropout(dropout, name='Dropout_' + str(i)))
        activation_list.append(Activation('relu', name='ReLU_' + str(i)))

    add_layer = Add(name='sum')

    return layers_list, bn_list_prop, dropout_list, activation_list, add_layer, bn_list_text


def make_embedder_with_layers(input_layer, layer_name, layers, dropout=0, bn=True, temporalBN=False):
    prev_layer = input_layer

    for i in range(1, len(layers)):

        if bn:
            if temporalBN:
                prev_layer = BatchNormalization(axis=-2, name=layer_name + '_TBN_' + str(i))(prev_layer)
            else:
                prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(i))(prev_layer)

        prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(i))(prev_layer)

        prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(i))(prev_layer)

        prev_layer = TimeDistributed(layers[i],
                                     name=layer_name + '_TD_' + str(i))(prev_layer)
    if bn:
        if temporalBN:
            prev_layer = BatchNormalization(axis=-2, name=layer_name + '_TBN_' + str(len(layers)))(prev_layer)
        else:
            prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(len(layers)))(prev_layer)

    prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(len(layers)))(prev_layer)

    prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(len(layers)))(prev_layer)

    prev_layer = TimeDistributed(layers[0],
                                 name=layer_name + '_TD_' + str(len(layers)))(prev_layer)

    prev_layer = Add(name=layer_name + '_sum')([input_layer, prev_layer])

    return prev_layer


def make_embedder_with_all_layers(input_layer, layer_name, layers, dropout=0, bn=True, temporalBN=False):
    prev_layer = input_layer

    bn_layers = layers[1]
    dropout_layers = layers[2]
    activation_layers = layers[3]
    add_layer = layers[4]
    bn_t_layers = layers[5]
    layers = layers[0]

    for i in range(1, len(layers)):

        if bn:
            if temporalBN:
                if layer_name == 'text':
                    prev_layer = bn_t_layers[i](prev_layer)
                else:
                    prev_layer = bn_layers[i](prev_layer)
            else:
                prev_layer = bn_layers[i](prev_layer)

        prev_layer = dropout_layers[i](prev_layer)

        prev_layer = activation_layers[i](prev_layer)

        prev_layer = TimeDistributed(layers[i],
                                     name=layer_name + '_TD_' + str(i))(prev_layer)
    if bn:
        if temporalBN:
            if layer_name == 'text':
                prev_layer = bn_t_layers[0](prev_layer)
            else:
                prev_layer = bn_layers[0](prev_layer)
        else:
            prev_layer = bn_layers[0](prev_layer)

    prev_layer = dropout_layers[0](prev_layer)

    prev_layer = activation_layers[0](prev_layer)

    prev_layer = TimeDistributed(layers[0],
                                 name=layer_name + '_TD_' + str(len(layers)))(prev_layer)

    prev_layer = add_layer([input_layer, prev_layer])

    return prev_layer


def build_net_2(bow=None,
                cross_embed=True,
                text_length=200, propos_length=75,
                regularizer_weight=0.001,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                embedding_size=int(25),
                embedder_layers=2,
                avg_pad=int(10),
                resnet_layers=(2, 2),
                res_size=50,
                final_size=int(20),
                outputs=(2, 5, 5, 5),
                bn_embed=True,
                bn_res=True,
                bn_final=True,
                single_LSTM=False,
                pooling=0):


    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="source_input_L")
        targetprop_il = Input(shape=(propos_length,), name="target_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="source_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="target_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="source_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="target_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    mark_il = Input(shape=(text_length, 2), name="mark_input_L")
    dist_il = Input(shape=(10,), name="dist_input_L")

    prev_text_l = Concatenate(name="mark_concatenation")([prev_text_l, mark_il])

    text_embed1 = make_deep_word_embedder(prev_text_l, 'text', regularizer_weight, layers=embedder_layers,
                                         dropout=dropout_embedder, layers_size=embedding_size, bn=bn_embed)
    source_embed1 = make_deep_word_embedder(prev_source_l, 'source', regularizer_weight, layers=embedder_layers,
                                               dropout=dropout_embedder, layers_size=embedding_size)
    target_embed1 = make_deep_word_embedder(prev_target_l, 'target', regularizer_weight, layers=embedder_layers,
                                               dropout=dropout_embedder, layers_size=embedding_size)

    if pooling > 0:
        text_embed1 = AveragePooling1D(pool_size=pooling, name='text_pooling')(text_embed1)
        prop_pooling = AveragePooling1D(pool_size=pooling, name='prop_pooling')
        source_embed1 = prop_pooling(source_embed1)
        target_embed1 = prop_pooling(target_embed1)

    text_embed1 = BatchNormalization()(text_embed1)
    source_embed1 = BatchNormalization()(source_embed1)
    target_embed1 = BatchNormalization()(target_embed1)

    if single_LSTM:
        embed2 = Bidirectional(LSTM(units=embedding_size,
                                    dropout=dropout_embedder,
                                    recurrent_dropout=dropout_embedder,
                                    kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                    recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                    bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                    return_sequences=False,
                                    unroll=False, # not possible to unroll if the time shape is not specified
                                    name='prop_LSTM',
                                    ),
                               merge_mode='mul',
                               name='biLSTM'
                               )

        source_embed2 = embed2(source_embed1)
        target_embed2 = embed2(target_embed1)

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='text_LSTM'),
                                    merge_mode='mul',
                                    name='text_biLSTM'
                                    )(text_embed1)
    else:

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False, # not possible to unroll if the time shape is not specified
                                       name='text_LSTM'),
                                  merge_mode='mul',
                                  name='text_biLSTM'
                                  )(text_embed1)

        source_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False,  # not possible to unroll if the time shape is not specified
                                       name='source_LSTM'),
                                  merge_mode='mul',
                                  name='source_biLSTM'
                                  )(source_embed1)

        target_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='target_LSTM'),
                                    merge_mode='mul',
                                  name='target_biLSTM'
                                    )(target_embed1)

    if cross_embed:
        text_embed1 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=True,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='text_joint_LSTM'),
                                      merge_mode='mul',
                                      name='text_joint_biLSTM'
                                      )(text_embed1)

        source_embed1 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=True,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='source_joint_LSTM'),
                                      merge_mode='mul',
                                      name='source_joint_biLSTM'
                                      )(source_embed1)


        target_embed1 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=True,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='target_joint_LSTM'),
                                      merge_mode='mul',
                                      name='target_joint_biLSTM'
                                      )(target_embed1)

        text_avg = AveragePooling1D(pool_size=avg_pad, name='text_avg')(text_embed1)
        source_avg = AveragePooling1D(pool_size=avg_pad, name='source_avg')(source_embed1)
        target_avg = AveragePooling1D(pool_size=avg_pad, name='target_avg')(target_embed1)

        text_flat = Flatten(name='text_flatten')(text_avg)
        source_flat = Flatten(name='source_flatten')(source_avg)
        target_flat = Flatten(name='target_flatten')(target_avg)

        prev_layer = Concatenate(name='word_embed_merge')([text_flat, source_flat, target_flat, dist_il])

        prev_layer = BatchNormalization(name='joint_BN')(prev_layer)

        prev_layer = Dropout(dropout_embedder, name='joint_Dropout')(prev_layer)

        prev_layer = Dense(units=embedding_size,
                            activation='relu',
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                            bias_regularizer=keras.regularizers.l2(regularizer_weight),
                            name='joint_dense'
                            )(prev_layer)

        text_embed2 = Concatenate(name='text_embed_concat')([prev_layer, text_embed2])
        source_embed2 = Concatenate(name='source_embed_concat')([prev_layer, source_embed2])
        target_embed2 = Concatenate(name='target_embed_concat')([prev_layer, target_embed2])


    prev_l = Concatenate(name='embed_merge')([text_embed2, source_embed2, target_embed2, dist_il])

    if bn_res:
        prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(prev_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet, bn=bn_res)

    if bn_final:
        prev_l = BatchNormalization(name='final_BN')(prev_l)

    link_ol = Dense(units=outputs[0],
                    name='link',
                    activation='softmax',
                    )(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il, dist_il, mark_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def build_net_4(bow=None,
                text_length=200, propos_length=75,
                regularizer_weight=0.001,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                dropout_final=0,
                embedding_size=int(25),
                embedder_layers=2,
                resnet_layers=(2, 2),
                res_size=50,
                final_size=int(20),
                outputs=(2, 5, 5, 5),
                bn_embed=True,
                bn_res=True,
                bn_final=True,
                single_LSTM=False,
                pooling=0,
                text_pooling=0,
                pooling_type='avg'):

    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="source_input_L")
        targetprop_il = Input(shape=(propos_length,), name="target_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="source_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="target_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="source_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="target_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    mark_il = Input(shape=(text_length, 2), name="mark_input_L")
    dist_il = Input(shape=(10,), name="dist_input_L")

    prev_text_l = Concatenate(name="mark_concatenation")([prev_text_l, mark_il])

    if embedder_layers > 0:
        prev_text_l = make_deep_word_embedder(prev_text_l, 'text', regularizer_weight, layers=embedder_layers,
                                             dropout=dropout_embedder, layers_size=embedding_size, bn=bn_embed)
        prev_source_l = make_deep_word_embedder(prev_source_l, 'source', regularizer_weight, layers=embedder_layers,
                                                   dropout=dropout_embedder, layers_size=embedding_size)
        prev_target_l = make_deep_word_embedder(prev_target_l, 'target', regularizer_weight, layers=embedder_layers,
                                                   dropout=dropout_embedder, layers_size=embedding_size)

    if pooling > 0:
        if not text_pooling > 0:
            text_pooling = pooling
        if pooling_type == 'max':
            pooling_class = MaxPool1D
        else:
            pooling_class = AveragePooling1D
            prev_text_l = pooling_class(pool_size=text_pooling, name='text_pooling')(prev_text_l)
        prop_pooling = pooling_class(pool_size=pooling, name='prop_pooling')
        prev_source_l = prop_pooling(prev_source_l)
        prev_target_l = prop_pooling(prev_target_l)

    text_embed1 = BatchNormalization()(prev_text_l)
    source_embed1 = BatchNormalization()(prev_source_l)
    target_embed1 = BatchNormalization()(prev_target_l)

    if single_LSTM:
        embed2 = Bidirectional(LSTM(units=embedding_size,
                                    dropout=dropout_embedder,
                                    recurrent_dropout=dropout_embedder,
                                    kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                    recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                    bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                    return_sequences=False,
                                    unroll=False, # not possible to unroll if the time shape is not specified
                                    name='prop_LSTM',
                                    ),
                               merge_mode='mul',
                               name='biLSTM'
                               )

        source_embed2 = embed2(source_embed1)
        target_embed2 = embed2(target_embed1)

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='text_LSTM'),
                                    merge_mode='mul',
                                    name='text_biLSTM'
                                    )(text_embed1)
    else:

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False, # not possible to unroll if the time shape is not specified
                                       name='text_LSTM'),
                                  merge_mode='mul',
                                  name='text_biLSTM'
                                  )(text_embed1)

        source_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False,  # not possible to unroll if the time shape is not specified
                                       name='source_LSTM'),
                                  merge_mode='mul',
                                  name='source_biLSTM'
                                  )(source_embed1)

        target_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='target_LSTM'),
                                    merge_mode='mul',
                                  name='target_biLSTM'
                                    )(target_embed1)

    prev_l = Concatenate(name='embed_merge')([text_embed2, source_embed2, target_embed2, dist_il])

    if bn_res:
        prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(prev_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet, bn=bn_res)

    if bn_final:
        prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_final, name='final_dropout')(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    rel_0 = Lambda(create_crop_fn(1, 0, 1))(rel_ol)
    rel_2 = Lambda(create_crop_fn(1, 2, 3))(rel_ol)
    rel_1 = Lambda(create_crop_fn(1, 1, 2))(rel_ol)
    rel_3 = Lambda(create_crop_fn(1, 3, 4))(rel_ol)
    rel_4 = Lambda(create_crop_fn(1, 4, 5))(rel_ol)

    pos_rel = Add()([rel_0, rel_2])
    neg_rel = Add()([rel_1, rel_3, rel_4])
    link_ol = Concatenate(name='link')([pos_rel, neg_rel])

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il, dist_il, mark_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def build_net_5(bow=None,
                text_length=200, propos_length=75,
                regularizer_weight=0.001,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                dropout_final=0,
                embedding_size=int(25),
                embedder_layers=2,
                resnet_layers=(2, 2),
                res_size=50,
                final_size=int(20),
                outputs=(2, 5, 5, 5),
                bn_embed=True,
                bn_res=True,
                bn_final=True,
                single_LSTM=False,
                pooling=0,
                text_pooling=0,
                pooling_type='avg'):

    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="source_input_L")
        targetprop_il = Input(shape=(propos_length,), name="target_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="source_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="target_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="source_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="target_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    mark_il = Input(shape=(text_length, 2), name="mark_input_L")
    dist_il = Input(shape=(10,), name="dist_input_L")

    shape = int(np.shape(prev_text_l)[2])
    dense_layers = make_embedder_layers(regularizer_weight, shape=shape, layers=embedder_layers,
                                        layers_size=embedding_size)

    if embedder_layers > 0:
        prev_text_l = make_embedder_with_layers(prev_text_l, 'text',
                                             dropout=dropout_embedder, layers=dense_layers, bn=bn_embed)
        prev_source_l = make_embedder_with_layers(prev_source_l, 'source',
                                                   dropout=dropout_embedder, layers=dense_layers, bn=bn_embed)
        prev_target_l = make_embedder_with_layers(prev_target_l, 'target',
                                                   dropout=dropout_embedder, layers=dense_layers, bn=bn_embed)

    prev_text_l = Concatenate(name="mark_concatenation")([prev_text_l, mark_il])

    if pooling > 0:
        if not text_pooling > 0:
            text_pooling = pooling
        if pooling_type == 'max':
            pooling_class = MaxPool1D
        else:
            pooling_class = AveragePooling1D
            prev_text_l = pooling_class(pool_size=text_pooling, name='text_pooling')(prev_text_l)
        prop_pooling = pooling_class(pool_size=pooling, name='prop_pooling')
        prev_source_l = prop_pooling(prev_source_l)
        prev_target_l = prop_pooling(prev_target_l)

    text_embed1 = BatchNormalization()(prev_text_l)
    source_embed1 = BatchNormalization()(prev_source_l)
    target_embed1 = BatchNormalization()(prev_target_l)

    if single_LSTM:
        embed2 = Bidirectional(LSTM(units=embedding_size,
                                    dropout=dropout_embedder,
                                    recurrent_dropout=dropout_embedder,
                                    kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                    recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                    bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                    return_sequences=False,
                                    unroll=False, # not possible to unroll if the time shape is not specified
                                    name='prop_LSTM',
                                    ),
                               merge_mode='mul',
                               name='biLSTM'
                               )

        source_embed2 = embed2(source_embed1)
        target_embed2 = embed2(target_embed1)

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='text_LSTM'),
                                    merge_mode='mul',
                                    name='text_biLSTM'
                                    )(text_embed1)
    else:

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False, # not possible to unroll if the time shape is not specified
                                       name='text_LSTM'),
                                  merge_mode='mul',
                                  name='text_biLSTM'
                                  )(text_embed1)

        source_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False,  # not possible to unroll if the time shape is not specified
                                       name='source_LSTM'),
                                      merge_mode='mul',
                                      name='source_biLSTM'
                                      )(source_embed1)

        target_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='target_LSTM'),
                                    merge_mode='mul',
                                  name='target_biLSTM'
                                    )(target_embed1)

    prev_l = Concatenate(name='embed_merge')([text_embed2, source_embed2, target_embed2, dist_il])

    if bn_res:
        prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(prev_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet, bn=bn_res)

    if bn_final:
        prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_final, name='final_dropout')(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    rel_0 = Lambda(create_crop_fn(1, 0, 1), name='rel0')(rel_ol)
    rel_2 = Lambda(create_crop_fn(1, 2, 3), name='rel2')(rel_ol)
    rel_1 = Lambda(create_crop_fn(1, 1, 2), name='rel1')(rel_ol)
    rel_3 = Lambda(create_crop_fn(1, 3, 4), name='rel3')(rel_ol)
    rel_4 = Lambda(create_crop_fn(1, 4, 5), name='rel4')(rel_ol)

    pos_rel = Add(name='rel_pos')([rel_0, rel_2])
    neg_rel = Add(name='rel_neg')([rel_1, rel_3, rel_4])
    link_ol = Concatenate(name='link')([pos_rel, neg_rel])

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il, dist_il, mark_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model

def build_net_7(bow=None,
                text_length=200, propos_length=75,
                regularizer_weight=0.001,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                dropout_final=0,
                embedding_size=int(25),
                embedder_layers=2,
                resnet_layers=(2, 2),
                res_size=50,
                final_size=int(20),
                outputs=(2, 5, 5, 5),
                bn_embed=True,
                bn_res=True,
                bn_final=True,
                single_LSTM=False,
                pooling=0,
                text_pooling=0,
                pooling_type='avg',
                same_DE_layers=False,
                context=True,
                distance=True,
                temporalBN=False):

    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="source_input_L")
        targetprop_il = Input(shape=(propos_length,), name="target_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="source_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="target_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="source_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="target_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    mark_il = Input(shape=(text_length, 2), name="mark_input_L")
    dist_il = Input(shape=(10,), name="dist_input_L")

    shape = int(np.shape(prev_text_l)[2])
    layers = make_embedder_layers(regularizer_weight, shape=shape, layers=embedder_layers,
                                                   layers_size=embedding_size, temporalBN=temporalBN)
    if same_DE_layers:
        make_embedder = make_embedder_with_all_layers
    else:
        make_embedder = make_embedder_with_layers
        layers = layers[0]

    if embedder_layers > 0:
        prev_text_l = make_embedder(prev_text_l, 'text', dropout=dropout_embedder, layers=layers,
                                                 bn=bn_embed, temporalBN=temporalBN)

        prev_source_l = make_embedder(prev_source_l, 'source', dropout=dropout_embedder,
                                                  layers=layers, bn=bn_embed, temporalBN=temporalBN)
        prev_target_l = make_embedder(prev_target_l, 'target', dropout=dropout_embedder,
                                                  layers=layers, bn=bn_embed, temporalBN=temporalBN)

    if same_DE_layers:
        if bn_embed:
            if temporalBN:
                bn_layer = BatchNormalization(name="TBN_DENSE_prop", axis=-2)
                bn_layer_t = BatchNormalization(name="TBN_DENSE_text", axis=-2)
            else:
                bn_layer = BatchNormalization(name="BN_DENSE_generic")
                bn_layer_t = bn_layer
            prev_text_l = bn_layer_t(prev_text_l)
            prev_source_l = bn_layer(prev_source_l)
            prev_target_l = bn_layer(prev_target_l)

        drop_layer = Dropout(dropout_embedder)

        prev_text_l = drop_layer(prev_text_l)
        prev_source_l = drop_layer(prev_source_l)
        prev_target_l = drop_layer(prev_target_l)

    else:
        if bn_embed:
            if temporalBN:
                prev_text_l = BatchNormalization(axis=-2)(prev_text_l)
                prev_source_l = BatchNormalization(axis=-2)(prev_source_l)
                prev_target_l = BatchNormalization(axis=-2)(prev_target_l)
            else:
                prev_text_l = BatchNormalization()(prev_text_l)
                prev_source_l = BatchNormalization()(prev_source_l)
                prev_target_l = BatchNormalization()(prev_target_l)

        prev_text_l = Dropout(dropout_embedder)(prev_text_l)
        prev_source_l = Dropout(dropout_embedder)(prev_source_l)
        prev_target_l = Dropout(dropout_embedder)(prev_target_l)

    relu_embedder = Dense(units=embedding_size,
                              activation='relu',
                              kernel_initializer='he_normal',
                              kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                              bias_regularizer=keras.regularizers.l2(regularizer_weight),
                              name='relu_embedder')

    prev_text_l = TimeDistributed(relu_embedder, name='TD_text_embedder')(prev_text_l)
    TD_prop = TimeDistributed(relu_embedder, name='TD_prop_embedder')
    prev_source_l = TD_prop(prev_source_l)
    prev_target_l = TD_prop(prev_target_l)

    prev_text_l = Concatenate(name="mark_concatenation")([prev_text_l, mark_il])

    if pooling > 0:
        if not text_pooling > 0:
            text_pooling = pooling
        if pooling_type == 'max':
            pooling_class = MaxPool1D
        else:
            pooling_class = AveragePooling1D
            prev_text_l = pooling_class(pool_size=text_pooling, name='text_pooling')(prev_text_l)
        prop_pooling = pooling_class(pool_size=pooling, name='prop_pooling')
        prev_source_l = prop_pooling(prev_source_l)
        prev_target_l = prop_pooling(prev_target_l)

    if bn_embed:
        if temporalBN:
            prev_text_l = BatchNormalization(name="TBN_LSTM_text", axis=-2)(prev_text_l)
        else:
            prev_text_l = BatchNormalization(name="BN_LSTM_text")(prev_text_l)

    if single_LSTM:
        if bn_embed:

            if temporalBN:
                bn_layer = BatchNormalization(name="TBN_LSTM_prop", axis=-2)
                prev_source_l = bn_layer(prev_source_l)
                prev_target_l = bn_layer(prev_target_l)
            else:
                bn_layer = BatchNormalization(name="BN_LSTM_prop")
                prev_source_l = bn_layer(prev_source_l)
                prev_target_l = bn_layer(prev_target_l)

        embed2 = Bidirectional(LSTM(units=embedding_size,
                                    dropout=dropout_embedder,
                                    recurrent_dropout=dropout_embedder,
                                    kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                    recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                    bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                    return_sequences=False,
                                    unroll=False, # not possible to unroll if the time shape is not specified
                                    name='prop_LSTM',
                                    ),
                               merge_mode='mul',
                               name='biLSTM'
                               )

        source_embed2 = embed2(prev_source_l)
        target_embed2 = embed2(prev_target_l)

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='text_LSTM'),
                                    merge_mode='mul',
                                    name='text_biLSTM'
                                    )(prev_text_l)
    else:
        if bn_embed:
            if temporalBN:
                prev_source_l = BatchNormalization(name="TBN_LSTM_source", axis=-2)(prev_source_l)
                prev_target_l = BatchNormalization(name="TBN_LSTM_target", axis=-2)(prev_target_l)
            else:
                prev_source_l = BatchNormalization(name="BN_LSTM_source")(prev_source_l)
                prev_target_l = BatchNormalization(name="BN_LSTM_target")(prev_target_l)

        text_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False, # not possible to unroll if the time shape is not specified
                                       name='text_LSTM'),
                                  merge_mode='mul',
                                  name='text_biLSTM'
                                  )(prev_text_l)

        source_embed2 = Bidirectional(LSTM(units=embedding_size,
                                       dropout=dropout_embedder,
                                       recurrent_dropout=dropout_embedder,
                                       kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                       recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                       bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                       return_sequences=False,
                                       unroll=False,  # not possible to unroll if the time shape is not specified
                                       name='source_LSTM'),
                                      merge_mode='mul',
                                      name='source_biLSTM'
                                      )(prev_source_l)

        target_embed2 = Bidirectional(LSTM(units=embedding_size,
                                         dropout=dropout_embedder,
                                         recurrent_dropout=dropout_embedder,
                                         kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                         recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                         bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                         return_sequences=False,
                                         unroll=False,  # not possible to unroll if the time shape is not specified
                                         name='target_LSTM'),
                                    merge_mode='mul',
                                  name='target_biLSTM'
                                    )(prev_target_l)

    if context and distance:
        prev_l = Concatenate(name='embed_merge')([text_embed2, source_embed2, target_embed2, dist_il])
    elif distance:
        prev_l = Concatenate(name='embed_merge')([source_embed2, target_embed2, dist_il])
    elif context:
        prev_l = Concatenate(name='embed_merge')([text_embed2, source_embed2, target_embed2])
    else:
        prev_l = Concatenate(name='embed_merge')([source_embed2, target_embed2])

    if bn_res:
        prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(prev_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet, bn=bn_res)

    if bn_final:
        prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_final, name='final_dropout')(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    rel_0 = Lambda(create_crop_fn(1, 0, 1), name='rel0')(rel_ol)
    rel_2 = Lambda(create_crop_fn(1, 2, 3), name='rel2')(rel_ol)
    rel_1 = Lambda(create_crop_fn(1, 1, 2), name='rel1')(rel_ol)
    rel_3 = Lambda(create_crop_fn(1, 3, 4), name='rel3')(rel_ol)
    rel_4 = Lambda(create_crop_fn(1, 4, 5), name='rel4')(rel_ol)

    pos_rel = Add(name='rel_pos')([rel_0, rel_2])
    neg_rel = Add(name='rel_neg')([rel_1, rel_3, rel_4])
    link_ol = Concatenate(name='link')([pos_rel, neg_rel])

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il, dist_il, mark_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def build_net_6(bow=None,
                text_length=200, propos_length=75,
                regularizer_weight=0.001,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                dropout_final=0,
                embedding_size=int(25),
                embedder_layers=2,
                resnet_layers=(2, 2),
                res_size=50,
                final_size=int(20),
                outputs=(2, 5, 5, 5),
                bn_embed=True,
                bn_res=True,
                bn_final=True,
                pooling_type='avg'):

    if bow is not None:
        text_il = Input(shape=(text_length,), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length,), name="source_input_L")
        targetprop_il = Input(shape=(propos_length,), name="target_input_L")

        prev_text_l = Embedding(bow.shape[0],
                                bow.shape[1],
                                weights=[bow],
                                input_length=text_length,
                                trainable=False,
                                name="text_embed")(text_il)

        prev_source_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="source_embed")(sourceprop_il)

        prev_target_l = Embedding(bow.shape[0],
                                  bow.shape[1],
                                  weights=[bow],
                                  input_length=propos_length,
                                  trainable=False,
                                  name="target_embed")(targetprop_il)
    else:
        text_il = Input(shape=(text_length, DIM), name="text_input_L")
        sourceprop_il = Input(shape=(propos_length, DIM), name="source_input_L")
        targetprop_il = Input(shape=(propos_length, DIM), name="target_input_L")
        prev_text_l = text_il
        prev_source_l = sourceprop_il
        prev_target_l = targetprop_il

    mark_il = Input(shape=(text_length, 2), name="mark_input_L")
    dist_il = Input(shape=(10,), name="dist_input_L")

    shape = int(np.shape(prev_text_l)[2])
    dense_layers = make_embedder_layers(regularizer_weight, shape=shape, layers=embedder_layers,
                                        layers_size=embedding_size)

    if embedder_layers > 0:
        prev_text_l = make_embedder_with_layers(prev_text_l, 'text',
                                             dropout=dropout_embedder, layers=dense_layers, bn=bn_embed)
        prev_source_l = make_embedder_with_layers(prev_source_l, 'source',
                                                   dropout=dropout_embedder, layers=dense_layers, bn=bn_embed)
        prev_target_l = make_embedder_with_layers(prev_target_l, 'target',
                                                   dropout=dropout_embedder, layers=dense_layers, bn=bn_embed)

    prev_text_l = Concatenate(name="mark_concatenation")([prev_text_l, mark_il])

    if bn_embed:
        prev_text_l = BatchNormalization()(prev_text_l)
        prev_source_l = BatchNormalization()(prev_source_l)
        prev_target_l = BatchNormalization()(prev_target_l)

    prev_text_l = Dropout(dropout_embedder)(prev_text_l)
    prev_source_l = Dropout(dropout_embedder)(prev_source_l)
    prev_target_l = Dropout(dropout_embedder)(prev_target_l)

    prev_text_l = TimeDistributed(Dense(units=embedding_size,
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                          bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                          name='relu_text_embedder'),
                                  name='TD_text_embedder')(prev_text_l)

    prev_source_l = TimeDistributed(Dense(units=embedding_size,
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                          bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                          name='relu_source_embedder'),
                                        name='TD_source_embedder')(prev_source_l)

    prev_target_l = TimeDistributed(Dense(units=embedding_size,
                                          activation='relu',
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                          bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                          name='relu_target_embedder'),
                                        name='TD_target_embedder')(prev_target_l)

    if pooling_type == 'max':
        pooling_class = GlobalMaxPooling1D
    else:
        pooling_class = GlobalAveragePooling1D
    prop_pooling = pooling_class(name='pooling')
    prev_source_l = prop_pooling(prev_source_l)
    prev_target_l = prop_pooling(prev_target_l)
    prev_text_l = prop_pooling(prev_text_l)

    prev_l = Concatenate(name='embed_merge')([prev_text_l, prev_source_l, prev_target_l, dist_il])

    if bn_res:
        prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(prev_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet, bn=bn_res)

    if bn_final:
        prev_l = BatchNormalization(name='final_BN')(prev_l)

    prev_l = Dropout(dropout_final, name='final_dropout')(prev_l)

    rel_ol = Dense(units=outputs[1],
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    rel_0 = Lambda(create_crop_fn(1, 0, 1), name='rel0')(rel_ol)
    rel_2 = Lambda(create_crop_fn(1, 2, 3), name='rel2')(rel_ol)
    rel_1 = Lambda(create_crop_fn(1, 1, 2), name='rel1')(rel_ol)
    rel_3 = Lambda(create_crop_fn(1, 3, 4), name='rel3')(rel_ol)
    rel_4 = Lambda(create_crop_fn(1, 4, 5), name='rel4')(rel_ol)

    pos_rel = Add(name='rel_pos')([rel_0, rel_2])
    neg_rel = Add(name='rel_neg')([rel_1, rel_3, rel_4])
    link_ol = Concatenate(name='link')([pos_rel, neg_rel])

    source_ol = Dense(units=outputs[2],
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=outputs[3],
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il, dist_il, mark_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def create_crop_fn(dimension, start, end):
    """
    From https://github.com/keras-team/keras/issues/890#issuecomment-319671916
    Crops (or slices) a Tensor on a given dimension from start to end
    example : to crop tensor x[:, :, 5:10]
    call slice(2, 5, 10) as you want to crop on the second dimension
    :param dimension: dimension of the object. The crop will be performed on the last dimension
    :param start: starting index
    :param end: ending index (excluded)
    :return:
    """
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    func.__name__ = "crop_" + str(dimension) + "_" + str(start) + "_" + str(end)
    return func


if __name__ == '__main__':

    bow = np.array([[0]*300]*50)

    """

    model = build_net_5(bow=None,
                        text_length=552, propos_length=153,
                        regularizer_weight=0.001,
                        dropout_embedder=0.1,
                        dropout_resnet=0.1,
                        dropout_final=0.1,
                        embedding_size=int(20),
                        embedder_layers=4,
                        resnet_layers=(1, 2),
                        res_size=5,
                        final_size=int(20),
                        bn_embed=True,
                        bn_res=True,
                        bn_final=True,
                        single_LSTM=True,
                        pooling=10,
                        text_pooling=50,
                        pooling_type='avg')

    plot_model(model, to_file='model5.png', show_shapes=True)

    model = build_net_7(bow=None,
                        text_length=552, propos_length=153,
                        regularizer_weight=0.001,
                        dropout_embedder=0.1,
                        dropout_resnet=0.1,
                        dropout_final=0.1,
                        embedding_size=int(20),
                        embedder_layers=4,
                        resnet_layers=(1, 2),
                        res_size=5,
                        final_size=int(20),
                        bn_embed=True,
                        bn_res=True,
                        bn_final=True,
                        single_LSTM=True,
                        pooling=10,
                        text_pooling=50,
                        pooling_type='avg')

    plot_model(model, to_file='model7.png', show_shapes=True)

    model = build_net_6(bow=None,
                        text_length=552, propos_length=153,
                        regularizer_weight=0.001,
                        dropout_embedder=0.1,
                        dropout_resnet=0.1,
                        dropout_final=0.1,
                        embedding_size=int(20),
                        embedder_layers=4,
                        resnet_layers=(1, 2),
                        res_size=5,
                        final_size=int(20),
                        bn_embed=True,
                        bn_res=True,
                        bn_final=True,
                        pooling_type='avg')

    plot_model(model, to_file='model6.png', show_shapes=True)
    """

    model = build_net_7(bow=bow,
                        text_length=552,
                        propos_length=153,
                        res_size=10,
                        resnet_layers=(1, 2),
                        embedding_size=20,
                        embedder_layers=2,
                        final_size=10,
                        regularizer_weight=0.0001,
                        dropout_resnet=0.1,
                        dropout_embedder=0.1,
                        dropout_final=0.1,
                        bn_embed=True,
                        bn_res=True,
                        bn_final=True,
                        single_LSTM=True,
                        pooling=10,
                        text_pooling=50,
                        temporalBN=True,
                        same_DE_layers=True,
                        context=True,
                        distance=True)

    plot_model(model, to_file='net7.png', show_shapes=True)

    print("YEP")