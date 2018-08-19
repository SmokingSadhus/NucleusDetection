import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
#sess = tf.Session()
import keras.backend as K
#K.set_session(sess)
import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json


lambda_obj = 1
lambda_noobj = 1

def custom_activation(x):
    #print(x.shape)
    #return tf.concat([tf.sigmoid(x[:,:,0:1]) , tf.nn.relu(x[:,:,1:5])],axis = 2)
    return tf.concat([tf.sigmoid(x[:,:,:,0:1]) , tf.nn.relu(x[:,:,:,1:5])],axis = 3)
    #tf.sigmoid(x[:,:,:,:,0:1])
    #tf.nn.relu(x[:,:,:,:,1:5])
    #return x
    

#def custom_activation(x):
#    return K.concatenate([K.sigmoid(x[:,:,0:1]) , K.relu(x[:,:,1:5])],axis = 2)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})


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
    #X = Activation('relu')(X)
    #X = Activation(custom_activation)(X)
    X = Activation('sigmoid')(X)

    model = Model(inputs = X_input, outputs = X, name='NucleusDetection')

    return model

#def yolo_loss(y_true, y_pred):
    #print(y_pred.shape)
    #print(y_true.shape)
#    class_error = tf.reduce_sum(tf.multiply((y_true[:,:,:,0]-y_pred[:,:,:,0]),(y_true[:,:,:,0]-y_pred[:,:,:,0])))
#    row_error = tf.reduce_sum(tf.multiply((y_true[:,:,:,1]-y_pred[:,:,:,1]),(y_true[:,:,:,1]-y_pred[:,:,:,1])))
#    col_error = tf.reduce_sum(tf.multiply((y_true[:,:,:,2]-y_pred[:,:,:,2]),(y_true[:,:,:,2]-y_pred[:,:,:,2])))
#    h_error = tf.reduce_sum(tf.abs(tf.sqrt(y_true[:,:,:,3])-tf.sqrt(y_pred[:,:,:,3])))
#    w_error = tf.reduce_sum(tf.abs(tf.sqrt(y_true[:,:,:,4])-tf.sqrt(y_pred[:,:,:,4])))
#    e1 = tf.add(class_error,row_error)
#    e2 = tf.add(e1,col_error)
#    e3 = tf.add(e2,h_error)
#    e4 = tf.add(e3,w_error)
    #print(e4.shape)
#    return e4/y_true.shape[0]

def yolo_loss(y_true, y_pred):
    #print(y_pred.shape)
    #print(y_true.shape)
    y_ret = tf.zeros([1,y_true.shape[0]])
    for i in range(0,int(y_true.shape[0])):
        op1 = y_true[i,:,:,:]
        op2 = y_pred[i,:,:,:]
        class_error = tf.reduce_sum(tf.multiply((op1[:,:,0]-op2[:,:,0]),(op1[:,:,0]-op2[:,:,0])))
        row_error = tf.reduce_sum(tf.multiply((op1[:,:,1]-op2[:,:,1]),(op1[:,:,1]-op2[:,:,1])))
        col_error = tf.reduce_sum(tf.multiply((op1[:,:,2]-op2[:,:,2]),(op1[:,:,2]-op2[:,:,2])))
        h_error = tf.reduce_sum(tf.abs(tf.sqrt(op1[:,:,3])-tf.sqrt(op2[:,:,3])))
        w_error = tf.reduce_sum(tf.abs(tf.sqrt(op1[:,:,4])-tf.sqrt(op2[:,:,4])))
        total_error = class_error + row_error + col_error + h_error + w_error
        y_ret[0,i] = total_error
    return y_ret



#def yolo_loss(y_true, y_pred):
#    y_new = y_true - y_pred
#    y_red = tf.reduce_sum(y_new,axis = -1)
#    print(y_red.shape)
#    return y_red


X_train = np.load('data_train.npy')
Y_train = np.load('labels_train.npy')
X_test = np.load('data_test.npy')
Y_test = np.load('labels_test.npy')

###################SizeReduction#############################

#X_train = X_train[0:1,:,:,:]
#Y_train = Y_train[0:1,:,:,:]
#X_test = X_test[0:1,:,:,:]
#Y_test = Y_test[0:1,:,:,:]

#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)
#print(Y_test.shape)

#############################################################


nucleus_model = model(X_train.shape[1:])

nucleus_model.compile(optimizer = 'adam',loss = 'mean_squared_error')

nucleus_model.fit(X_train, Y_train, epochs=50, batch_size=25)

#nucleus_model.fit_generator(X_train, Y_train, epochs=1, batch_size=25)

#preds = nucleus_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)

#print()
#print ("Loss = " + str(preds))

#print(X_test.shape)

o_p = nucleus_model.predict(X_test)

#o_p = nucleus_model.predict_generator(X_test)

#321`print(o_p.shape)

print(nucleus_model.summary())

nucleus_model_json = nucleus_model.to_json()
with open("nucleus_model.json", "w") as json_file:
    json_file.write(nucleus_model_json)
# serialize weights to HDF5
nucleus_model.save_weights("nucleus_model.h5")

print("Saved model to disk")

exit()
 
# later...
 
# load json and create model
json_file = open('nucleus_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
nucleus_model_loaded = model_from_json(loaded_model_json)
# load weights into new model
nucleus_model_loaded.load_weights("nucleus_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
nucleus_model_loaded.compile(loss=yolo_loss, optimizer='adam')







    
    

    

    
    
