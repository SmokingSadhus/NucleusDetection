import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D


def model(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((1, 1))(X_input)
    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu0')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu1')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu2')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu3')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu3')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    
    

    

    
    
