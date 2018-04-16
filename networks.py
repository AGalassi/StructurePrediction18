__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"


import keras
import numpy as np
from keras.layers import (BatchNormalization, Dropout, Dense, Input, Activation, LSTM, Conv1D, Add,
                          Bidirectional, Concatenate, Flatten, Embedding, TimeDistributed, AveragePooling1D)
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


def make_resnet(input_layer, regularizer_weight, layers=(2, 2), res_size=int(DIM/3)*3, dropout=0):
    prev_layer = input_layer
    prev_block = prev_layer
    blocks = layers[0]
    res_layers = layers[1]

    shape = int(np.shape(input_layer)[1])

    for i in range(1, blocks + 1):
        for j in range(1, res_layers):
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
                            layers=2, layers_size=int(DIM/10), dropout=0):
    prev_layer = input_layer

    shape = int(np.shape(input_layer)[2])
    for i in range(1, layers):

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
                outputs=(2, 5, 5, 5)):

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
                                         dropout=dropout_embedder, layers_size=embedding_size)
    source_embed1 = make_deep_word_embedder(prev_source_l, 'source', regularizer_weight, layers=embedder_layers,
                                               dropout=dropout_embedder, layers_size=embedding_size)
    target_embed1 = make_deep_word_embedder(prev_target_l, 'target', regularizer_weight, layers=embedder_layers,
                                               dropout=dropout_embedder, layers_size=embedding_size)

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

    prev_l = Dropout(dropout_resnet, name='merge_Dropout')(prev_l)

    prev_l = BatchNormalization(name='merge_BN')(prev_l)

    prev_l = Dense(units=final_size,
                   activation='relu',
                   kernel_initializer='he_normal',
                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                   name='merge_dense'
                   )(prev_l)

    prev_l = make_resnet(prev_l, regularizer_weight, resnet_layers,
                         res_size=res_size, dropout=dropout_resnet)

    prev_l = prev_l = BatchNormalization(name='final_BN')(prev_l)

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


if __name__ == '__main__':

    bow = np.array([[0]*300]*50)

    model = build_net_2(bow=bow, cross_embed=False)

    plot_model(model, to_file='model2.png', show_shapes=True)

    model = build_net_3(bow=bow)

    plot_model(model, to_file='model3.png', show_shapes=True)
