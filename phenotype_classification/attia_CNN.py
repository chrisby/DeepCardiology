#!/usr/bin/env python3
#
# Keras implementation of CNN as described by Attia et al. in 
# https://www.nature.com/articles/s41591-018-0240-2.
# Adjusted to work for binary classification tasks. 
# 
# Aspects that were not specified in the paper are marked as such.
#
# Author: Christian Bock License: MIT
#
# Important: Be aware that data_format='channels_first' was used.
# Check your ~/.keras/.keras.json do adjust to this.

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, Dropout, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization

import logging

filter_numbers      = [16, 16, 32, 32, 64, 64]
kernel_widths       = [5, 5, 5, 3, 3, 3]
pool_sizes          = [2, 2, 4, 2, 2, 4]

spatial_kernel_size = (12, 1) 
spatial_num_filters = 64    # Not specified in paper
dropout_rate        = 0.3   # Not specified in paper

dense_units         = [64, 32]

def build_cnn(input_shape: tuple, num_classes: int):
    input_layer = Input(shape=input_shape)
    last_layer = input_layer
    
    logging.info("Building temporal Conv layers")
    for i in range(len(pool_sizes)):
        temp_layer = get_temporal_layer(filter_numbers[i], kernel_widths[i],
                                        pool_sizes[i], last_layer)
        last_layer = temp_layer
    
    last_layer = get_spatial_layer(spatial_kernel_size, last_layer)
    last_layer = Flatten()(last_layer)

    logging.info("Building spatial Conv layers")
    for i in range(len(dense_units)):
        dense_layer = get_fully_connected_layer(dense_units[i], last_layer)
        last_layer = dense_layer

    output_layer = Dense(num_classes, activation='softmax')(last_layer)
    return Models(inputs=input_layer, outputs=output_layer)


def get_temporal_layer(N: int, k: int, p: int, input_layer):
    c = Conv2D(N, (1, k), padding='same')(input_layer)
    b = BatchNormalization(axis=1)(c) # Axis=1 as data_format = channels_first
    a = Activation('relu')(b)
    p = MaxPooling2D(pool_size=(1, p), data_format='channels_first')(a)
    return p

def get_spatial_layer(kernel_size: tuple, input_layer):
    c = Conv2D(spatial_num_filters, kernel_size)(input_layer)
    b = BatchNormalization(axis=1)(c)
    a = Activation('relu')(b)
    return a

def get_fully_connected_layer(units: int, input_layer):
    d = Dense(units)(input_layer)
    b = BatchNormalization(axis=1)(d)
    a = Activation('relu')(b)
    do = Dropout(dropout_rate)(a)
    return do
