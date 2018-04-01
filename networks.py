__author__ = "Andrea Galassi"
__copyright__ = "Copyright 2018, Andrea Galassi"
__license__ = "BSD 3-clause"
__version__ = "0.0.1"
__email__ = "a.galassi@unibo.it"




import keras
from keras.layers import (BatchNormalization, Dropout, Dense, Input, Activation, LSTM, Conv1D, Add,
                          Bidirectional, Concatenate, Flatten)
from keras.utils.vis_utils import plot_model
import pydot

DIM = 300


def build_net_1(text_length=200, propos_length=100,
                regularizer_weight=0.001,
                embedder_layers=2, resnet_layers=(3, 2)):

    text_il = Input(shape=(text_length, DIM), name="text_input_L")
    sourceprop_il = Input(shape=(propos_length, DIM), name="sourceprop_input_L")
    targetprop_il = Input(shape=(propos_length, DIM), name="targetprop_input_L")

    text_LSTM = make_embedder(text_il, 'text', regularizer_weight, embedder_layers)
    sourceprop_LSTM = make_embedder(sourceprop_il, 'sourceprop', regularizer_weight, embedder_layers)
    targetprop_LSTM = make_embedder(targetprop_il, 'targetprop', regularizer_weight, embedder_layers)

    concat_l = Concatenate(name= 'embed_merge')([text_LSTM, sourceprop_LSTM, targetprop_LSTM])

    prev_l = make_resnet(concat_l, regularizer_weight, resnet_layers)

    link_ol = Dense(units=2,
                    name='link',
                    activation='softmax',
                    )(prev_l)

    rel_ol = Dense(units=5,
                    name='relation',
                    activation='softmax',
                    )(prev_l)

    source_ol = Dense(units=5,
                      name='source',
                      activation='softmax',
                      )(prev_l)

    target_ol = Dense(units=5,
                      name='target',
                      activation='softmax',
                      )(prev_l)

    full_model = keras.Model(inputs=(text_il, sourceprop_il, targetprop_il),
                             outputs=(link_ol, rel_ol, source_ol, target_ol),
                             )

    return full_model


def make_resnet(input_layer, regularizer_weight, layers=(2, 2)):
    prev_layer = input_layer
    prev_block = prev_layer
    blocks = layers[0]
    res_layers = layers[1]

    for i in range(1, blocks + 1):
        for j in range(1, res_layers):
            prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(j))(prev_layer)

            prev_layer = Dense(units=int(DIM / 3),
                               activation=None,
                               kernel_initializer='he_normal',
                               kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                               bias_regularizer=keras.regularizers.l2(regularizer_weight),
                               name='resnet_dense_' + str(i) + '_' + str(j)
                               )(prev_layer)

        prev_layer = Activation('relu', name='resnet_ReLU_' + str(i) + '_' + str(res_layers))(prev_layer)

        prev_layer = Dense(units=int(DIM),
                           activation=None,
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                           bias_regularizer=keras.regularizers.l2(regularizer_weight),
                           name='resnet_dense_' + str(i) + '_' + str(res_layers)
                           )(prev_layer)

        prev_layer = Add(name='resnet_sum' + str(i))([prev_block, prev_layer])
        prev_block = prev_layer

    return prev_block


def make_embedder(input_layer, layer_name, regularizer_weight, layers=2):
    prev_layer = input_layer
    for i in range(1, layers):
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

    text_add = Add(name=layer_name + '_sum')([input_layer, prev_layer])

    text_LSTM = Bidirectional(LSTM(units=int(DIM/3),
                                   kernel_regularizer=keras.regularizers.l2(regularizer_weight),
                                   recurrent_regularizer=keras.regularizers.l2(regularizer_weight),
                                   bias_regularizer=keras.regularizers.l2(regularizer_weight),
                                   return_sequences=False,
                                   unroll=False, # not possible to unroll if the time shape is not specified
                                   name=layer_name + '_LSTM'),
                              merge_mode='mul',
                              )(text_add)

    return text_LSTM


if __name__ == '__main__':
    model = build_net_1()

    plot_model(model, to_file='model.png', show_shapes=True)

