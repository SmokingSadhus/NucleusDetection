import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
# sess = tf.Session()
import keras.backend as K
# K.set_session(sess)
import numpy as np
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Lambda
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json

lambda_obj = 1
lambda_noobj = 1

GRID_H = 19
GRID_W = 19
BOX = 1
CLASS = 1

BATCH_SIZE = 25

#ANCHORS = [0.57273, 0.677385]

ANCHORS = [1 , 1]

NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
COORD_SCALE = 1.0
CLASS_SCALE = 1.0
WARM_UP_BATCHES = 0
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
TRUE_BOX_BUFFER = 50


def custom_activation(x):
    # print(x.shape)
    # return tf.concat([tf.sigmoid(x[:,:,0:1]) , tf.nn.relu(x[:,:,1:5])],axis = 2)
    return tf.concat([tf.sigmoid(x[:, :, :, 0:1]) , tf.nn.relu(x[:, :, :, 1:5])], axis=3)
    # tf.sigmoid(x[:,:,:,:,0:1])
    # tf.nn.relu(x[:,:,:,:,1:5])
    # return x

# def custom_activation(x):
#    return K.concatenate([K.sigmoid(x[:,:,0:1]) , K.relu(x[:,:,1:5])],axis = 2)


get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def yolo_loss(y_true, y_pred):
    # print(y_pred.shape)
    # print(y_true.shape)
    y_ret = tf.zeros([1, y_true.shape[0]])
    for i in range(0, int(y_true.shape[0])):
        op1 = y_true[i, :, :, :]
        op2 = y_pred[i, :, :, :]
        class_error = tf.reduce_sum(tf.multiply((op1[:, :, 0] - op2[:, :, 0]), (op1[:, :, 0] - op2[:, :, 0])))
        row_error = tf.reduce_sum(tf.multiply((op1[:, :, 1] - op2[:, :, 1]), (op1[:, :, 1] - op2[:, :, 1])))
        col_error = tf.reduce_sum(tf.multiply((op1[:, :, 2] - op2[:, :, 2]), (op1[:, :, 2] - op2[:, :, 2])))
        h_error = tf.reduce_sum(tf.abs(tf.sqrt(op1[:, :, 3]) - tf.sqrt(op2[:, :, 3])))
        w_error = tf.reduce_sum(tf.abs(tf.sqrt(op1[:, :, 4]) - tf.sqrt(op2[:, :, 4])))
        total_error = class_error + row_error + col_error + h_error + w_error
        y_ret[0, i] = total_error
    return y_ret



# def yolo_loss(y_true, y_pred):
#    y_new = y_true - y_pred
#    y_red = tf.reduce_sum(y_new,axis = -1)
#    print(y_red.shape)
#    return y_red


X_train = np.load('data_train.npy')
Y_train = np.load('labels_train.npy')
true_boxes_train = np.load('True_Boxes.npy')
X_test = np.load('data_test.npy')
Y_test = np.load('labels_test.npy')

###################SizeReduction#############################

# X_train = X_train[0:1,:,:,:]
# Y_train = Y_train[0:1,:,:,:]
# X_test = X_test[0:1,:,:,:]
# Y_test = Y_test[0:1,:,:,:]

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

#############################################################

X_input = Input(X_train.shape[1:])
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

X = ZeroPadding2D((1, 1))(X_input)
X = Conv2D(16, (3, 3), strides=(1, 1), name='conv0')(X)
X = BatchNormalization(axis=3, name='bn0')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool0')(X)

X = ZeroPadding2D((1, 1))(X)
X = Conv2D(32, (3, 3), strides=(1, 1), name='conv1')(X)
X = BatchNormalization(axis=3, name='bn1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool1')(X)

X = ZeroPadding2D((1, 1))(X)
X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2')(X)
X = BatchNormalization(axis=3, name='bn2')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool2')(X)
    
X = ZeroPadding2D((1, 1))(X)
X = Conv2D(128, (3, 3), strides=(1, 1), name='conv3')(X)
X = BatchNormalization(axis=3, name='bn3')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool3')(X)

X = ZeroPadding2D((1, 1))(X)
X = Conv2D(128, (3, 3), strides=(1, 1), name='conv4')(X)
X = BatchNormalization(axis=3, name='bn4')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool4')(X)

X = Conv2D(6, (1, 1), strides=(1, 1), name='conv5')(X)
#X = Activation('relu')(X)
# X = Activation(custom_activation)(X)
X = Activation('sigmoid')(X)
output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS), name='FinalReshape')(X)
    
output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model([X_input, true_boxes], output)

# Code used from: 
#https://github.com/experiencor/keras-yolo2/blob/master/Yolo%20Step-by-Step.ipynb
def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]
    
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
    
    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)
    
    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)
    
    """
    Adjust prediction
    """
    # ## adjust x and y      
    #pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    pred_box_xy = tf.sigmoid(y_pred[..., :2])
    
    # ## adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
    
    # ## adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
    # ## adjust class probabilities
    pred_box_class = y_pred[..., 5:]
    
    """
    Adjust ground truth
    """
    # ## adjust x and y
    true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
    
    # ## adjust w and h
    true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically
    
    # ## adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half
    
    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half       
    
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)
    
    true_box_conf = iou_scores * y_true[..., 4]
    
    # ## adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    
    """
    Determine the masks
    """
    # ## coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
    
    # ## confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]
    
    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half    
    
    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)
    
    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half    
    
    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    


    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
    
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    
    # ## class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
    
    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE / 2.)
    seen = tf.assign_add(seen, 1.)
    
    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES),
                          #lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                          lambda: [true_box_xy + (0.5) * no_boxes_mask,
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2]) * no_boxes_mask,
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy,
                                   true_box_wh,
                                   coord_mask])
    
    """
    Finalize the loss
    """
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
    loss = loss_xy + loss_wh + loss_conf + loss_class
    
    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    """
    Debugging code
    """    
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall) 

    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
    
    return loss


nucleus_model = model

nucleus_model.compile(optimizer='adam', loss=custom_loss)

nucleus_model.fit([X_train, true_boxes_train], Y_train, epochs=50, batch_size=BATCH_SIZE)


# preds = nucleus_model.evaluate(X_test, Y_test, batch_size=32, verbose=1, sample_weight=None)


# print ("Loss = " + str(preds))

# print(X_test.shape)

#o_p = nucleus_model.predict(X_test)



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
    
