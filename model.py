import tensorflow as tf
import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

lambda_obj = 1
lambda_noobj = 1


def model(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((1, 1))(X_input)
    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(64, (3, 3), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides = (1, 1), name = 'conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name='max_pool4')(X)

    X = Conv2D(5, (1, 1), strides = (1, 1), name = 'conv5')(X)
    X = Activation('relu')(X)

    model = Model(inputs = X_input, outputs = X, name='NucleusDetection')

    return model

def yolo_loss(y_true, y_pred):
    class_error = tf.reduce_sum(tf.multiply((y_true[:,:,0]-y_pred[:,:,0]),(y_true[:,:,0]-y_pred[:,:,0])))
    row_error = tf.reduce_sum(tf.multiply((y_true[:,:,1]-y_pred[:,:,1]),(y_true[:,:,1]-y_pred[:,:,1])))
    col_error = tf.reduce_sum(tf.multiply((y_true[:,:,2]-y_pred[:,:,2]),(y_true[:,:,2]-y_pred[:,:,2])))
    h_error = tf.reduce_sum(tf.sqrt(y_true[:,:,3])-tf.sqrt(y_pred[:,:,3]))
    w_error = tf.reduce_sum(tf.sqrt(y_true[:,:,4])-tf.sqrt(y_pred[:,:,4]))
    e1 = tf.add(class_error,row_error)
    e2 = tf.add(e1,col_error)
    e3 = tf.add(e2,h_error)
    e4 = tf.add(e3,w_error)
    return e4
    
    #y_true[:,:,0] - y_pred[:,:,0]
    
    

X_train = np.load('data_train.npy')
Y_train = np.load('labels_train.npy')
X_test = np.load('data_test.npy')
Y_test = np.load('labels_test.npy')

nucleus_model = model(X_train.shape[1:])

nucleus_model.compile('adam',yolo_loss, metrics=['accuracy'])

nucleus_model.fit(X_train, Y_train, epochs=10, batch_size=50)

preds = nucleus_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))







    
    

    

    
    
