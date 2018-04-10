__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import keras
import numpy as np
from keras.layers import (BatchNormalization, Dropout, Dense, Input, Activation, LSTM, Conv1D, Add,
                          Bidirectional, Concatenate, Flatten, Embedding)
from keras.utils.vis_utils import plot_model
import pydot

DIM = 300


def build_net_1(bow=None,
                text_length=200, propos_length=100,
                regularizer_weight=0.001,
                dropout_embedder=0.1,
                dropout_resnet=0.1,
                embedding_size=int(DIM/3),
                embedder_layers=2,
                resnet_layers=(3, 2)):

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
                              embedding_size=embedding_size, dropout=dropout_embedder)
    sourceprop_LSTM = make_embedder(prev_source_l, 'sourceprop', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder)
    targetprop_LSTM = make_embedder(prev_target_l, 'targetprop', regularizer_weight, embedder_layers,
                              embedding_size=embedding_size, dropout=dropout_embedder)

    concat_l = Concatenate(name='embed_merge')([text_LSTM, sourceprop_LSTM, targetprop_LSTM])

    prev_l = make_resnet(concat_l, regularizer_weight, resnet_layers,
                         res_size=embedding_size*3, dropout=dropout_resnet)

    prev_l = BatchNormalization(name='final_BN')(prev_l)

    link_ol = Dense(units=2,
                    name='link',
                    activation='softmax',
                    )(prev_l)

    rel_ol = Dense(units=5,
                   name='relation',
                   activation='softmax',
                   )(prev_l)

    source_ol = Dense(units=3,
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=3,
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def make_resnet(input_layer, regularizer_weight, layers=(2, 2), res_size=int(DIM/3)*3, dropout=0):
    prev_layer = input_layer
    prev_block = prev_layer
    blocks = layers[0]
    res_layers = layers[1]

    for i in range(1, blocks + 1):
        for j in range(1, res_layers):
            prev_layer = BatchNormalization(name='resent_BN_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dropout(dropout, name='resnet_Dropout_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dense(units=int(DIM / 3),
                               activation=None,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                               bias_regularizer=keras.regularizers.l2(regularizer_weight),
                               name='resnet_dense_' + str(i) + '_' + str(j)
                               )(prev_layer)

        prev_layer = BatchNormalization(name='BN_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dropout(dropout, name='resnet_Dropout_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dense(units=res_size,
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
                  layers=2, embedding_size=int(DIM/3), dropout=0):
    prev_layer = input_layer
    for i in range(1, layers):

        prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(i))(prev_layer)

        prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(i))(prev_layer)

        prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(i))(prev_layer)

        prev_layer = Conv1D(filters=int(DIM / 10),
                            kernel_size=3,
                            padding='same',
                            activation=None,
                            kernel_initializer='he_normal',
                            kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                            bias_regularizer=keras.regularizers.l2(regularizer_weight),
                            name=layer_name + '_conv_' + str(i)
                            )(prev_layer)

    prev_layer = BatchNormalization(name=layer_name + '_BN_' + str(layers))(prev_layer)

    prev_layer = Dropout(dropout, name=layer_name + '_Dropout_' + str(layers))(prev_layer)

    prev_layer = Activation('relu', name=layer_name + '_ReLU_' + str(layers))(prev_layer)

    prev_layer = Conv1D(filters=int(DIM),
                        kernel_size=3,
                        padding='same',
                        activation=None,
                        kernel_initializer='he_normal',
                        kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                        bias_regularizer=keras.regularizers.l2(regularizer_weight),
                        name=layer_name + '_conv_' + str(layers)
                        )(prev_layer)

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


if __name__ == '__main__':
    model = build_net_1()

    plot_model(model, to_file='model.png', show_shapes=True)

