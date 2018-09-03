import csv
import os

from PIL import Image
from keras.layers import Activation
from keras.models import model_from_json
from keras.utils.generic_utils import get_custom_objects
#from model import custom_loss

import numpy as np
import tensorflow as tf

import time

#################
# image = Image.open("D:\Kaggle\extendedTestCrop.png")
# print(image.size)
# image_data = np.array(image, dtype='float32')
# print(image_data.shape)
# exit()
#############################
# rootdir = 'D:\Kaggle\stage1_test'
# rootdir = 'D:\Kaggle\stage1testTemp'

# output = 'output_stage_1.csv'

rootdir = 'D:\Kaggle\stage2_test_final'

output = 'output_stage_2.csv'

img_shape = 608

grid_size = 19

b_box_size = (img_shape / grid_size)

# b_box_size converted to box_scale for getting lower value coordinates for centers of objects row_c, col_c 
box_scale = 1

no_of_imgs = len(os.listdir(rootdir))

tst_img = np.zeros((1, img_shape, img_shape, 3))

TRUE_BOX_BUFFER = 50

GRID_H = 19
GRID_W = 19
BATCH_SIZE = 25
ANCHORS = [1 , 1]
BOX = 1
from keras.layers import Input
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))
COORD_SCALE = 1.0
CLASS_SCALE = 1.0
NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
WARM_UP_BATCHES = 0
CLASS = 1
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')

def preprocess_image_and_retSize(img_path, model_image_size):
    # image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    w, h = image.size
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    # resized_image = image.resize(model_image_size)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    # image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = image_data[:, :, 0:3]
    return image, image_data, h, w


# Not a good method for eliminating boxes, as we remove one of the 2 intersecting boxes(box with lower confidence). Use method in coursera deep learning course instead.
def remove_intersecting_boxes(orig_image_arr_list, confidence_list):
    pixel_to_img = dict()
    count = 0
    list_of_removables = []
    for orig_image in orig_image_arr_list:
        for i in range (orig_image.shape[0]):
            for j in range(orig_image.shape[1]):
                if orig_image[i][j][0] == 255:
                    key = ((j * orig_image.shape[0]) + i)
                    if key in pixel_to_img:
                        old = pixel_to_img[key]
                        if confidence_list[old] >= confidence_list[count]:
                            list_of_removables.append(count)
                            # Need to fond a way to break here
                        else:
                            list_of_removables.append(old)
                            pixel_to_img[key] = count
                    else:
                        pixel_to_img[key] = count
        count = count + 1
    ret_list = []
    for idx, val in enumerate(orig_image_arr_list):
        if idx not in list_of_removables:
            ret_list.append(val)
    return ret_list


def remove_intersecting_boxes_with_non_max_supression(boxes, confidence_list):
    return non_max_suppression(boxes, confidence_list)


def non_max_suppression(boxes, confidence_list):
    tuple_list = []
    for i in range(len(boxes)):
        tuple_list.append((i, boxes[i], (-1 * confidence_list[i])))
    tuple_list.sort(key=lambda tup: tup[2])
    mark_for_delete = [1] * len(tuple_list)
    
    for i in range(len(tuple_list) - 1):
        if(mark_for_delete[tuple_list[i][0]] != 0):
            for j in range(i + 1, len(tuple_list)):
                if(mark_for_delete[tuple_list[j][0]] != 0):
                    if(intersects(tuple_list[i][1], tuple_list[j][1])):
                        mark_for_delete[tuple_list[j][0]] = 0
    delete_indices = [i for i, e in enumerate(mark_for_delete) if e == 0]
    box_rem = np.delete(boxes, delete_indices, axis=0)
    return box_rem
                    
def validateValues(box1):
    if box1[2] < box1[0] or box1[3] < box1[1]:
        print ('error in box dimensions')


def intersects(box1, box2):
    y_top_left = max(box1[0] , box2[0])
    y_bottom_right = min(box1[2] , box2[2])
    x_top_left = max(box1[1] , box2[1])
    x_bottom_right = min(box1[3] , box2[3])
    validateValues(box1)
    validateValues(box2)
    if ((x_top_left > x_bottom_right) or (y_top_left > y_bottom_right)):
        return False
    else:
        return True
    
     


#def intersects(box1, box2):
#    validateValues(box1)
#    validateValues(box2)
#    l1x = box1[1]
#    r1x = box1[3]
#    l2x = box2[1]
#    r2x = box2[3]
#    l1y = box1[0]
#    r1y = box1[2]
#    l2y = box2[0]
#    r2y = box2[2]
#    if (l1x > r2x or l2x > r1x):
#        return False
#    if (l1y > r2y or l2y > r1y):
#        return False
#    return True

# bool doOverlap(Point l1, Point r1, Point l2, Point r2)
# {
#    // If one rectangle is on left side of other
#    if (l1.x > r2.x || l2.x > r1.x)
#        return false;
 
#    // If one rectangle is above other
#    if (l1.y < r2.y || l2.y < r1.y)
#        return false;
 
#    return true;
# }
  

def convert_to_rle_after_non_max(rem_box, rows, cols):
    row_first = rem_box[0]
    col_first = rem_box[1]
    row_last = rem_box[2]
    col_last = rem_box[3]
    # height_of_box = row_last - row_first+1
    height_of_box = row_last - row_first
    rle = []
    if(height_of_box >= rows or row_last >= rows or row_first >= rows):
        print('Massive error')
    for j in range(col_first, col_last + 1):
        pixel_st = (j * rows) + row_first + 1
        rle.append(pixel_st)
        rle.append(height_of_box)
    return rle


def write_to_csv(rle_op):
    with open(output, 'w') as csvfile:
        fieldnames = ['ImageId', 'EncodedPixels']
        writer = csv.DictWriter(csvfile, lineterminator='\n' , fieldnames=fieldnames)
        writer.writeheader()
        for image_id in rle_op.keys():
            list_of_masks = rle_op[image_id]
            if len(list_of_masks) == 0:
                writer.writerow({'ImageId':image_id, 'EncodedPixels':''})
            for mask in list_of_masks:
                if mask:
                    mask_str = ''
                    for pix_dist in mask:
                        mask_str = mask_str + str(pix_dist) + ' '
                    writer.writerow({'ImageId':image_id, 'EncodedPixels':mask_str})


def custom_activation(x):
    # print(x.shape)
    # return tf.concat([tf.sigmoid(x[:,:,0:1]) , tf.nn.relu(x[:,:,1:5])],axis = 2)
    return tf.concat([tf.sigmoid(x[:, :, :, 0:1]) , tf.nn.relu(x[:, :, :, 1:5])], axis=3)


# r_top, c_top, r_top + h_actual, c_top + w_actual
def reshape(old_values, new_width, new_height, old_width, old_height):
    r_top = old_values[0]
    c_top = old_values[1]
    r_bottom = old_values[2]
    c_bottom = old_values[3]
    
    row_conversion_factor = (new_height / old_height)
    col_conversion_factor = (new_width / old_width)
    
    r_top_new = int( r_top * row_conversion_factor )
    c_top_new = int( c_top * col_conversion_factor )
    r_bottom_new = int( r_bottom * row_conversion_factor )
    c_bottom_new = int( c_bottom * col_conversion_factor )
    if(r_bottom_new >= new_height or c_bottom_new >= new_width):
        print('Problems here')
        exit
    return [r_top_new, c_top_new, r_bottom_new, c_bottom_new]


    
    
def convertToNumpy(retrieved_img_list):
    np_array = np.zeros((len(retrieved_img_list), 4) , dtype=np.int)
    ct = 0
    for tup in retrieved_img_list:
        np_array[ct, :] = tup
        ct = ct + 1
    return np_array


get_custom_objects().update({'custom_activation': Activation(custom_activation)})

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

#Error detection code for intersection function
def checkForDuplicates(rle_op):
    for imd_id in rle_op:
        common = set()
        list_of_rles = rle_op[imd_id]
        for rle in list_of_rles:
            for st,leng in pairwise(rle):
                if st < 0 or leng < 0 :
                    print('start or length are lesser than zero')
                for j in range(st, st+leng):
                    if j in common:
                        print('Error in: ' + imd_id)
                    else:
                        common.add(j)


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
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
    #pred_box_xy = tf.sigmoid(y_pred[..., :2])
    
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
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
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



###### Code starts from here #######################################

json_file = open('nucleus_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
nucleus_model_loaded = model_from_json(loaded_model_json)
# load weights into new model
nucleus_model_loaded.load_weights("nucleus_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
nucleus_model_loaded.compile(loss=custom_loss, optimizer='adam')
#nucleus_model_loaded.compile(loss='mean_squared_error', optimizer='adam')

nucleus_model = nucleus_model_loaded

rle_op = dict()

for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\' + subdir
    # print(img_count)
    img_dir = os.listdir(rootdir_2)[0]
    img_file_name = os.listdir(rootdir_2 + '\\' + img_dir)[0]
    # img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + img_file_name
    print("Starting with " + img_file_name)
    st1 = time.time()
    img, img_preproc, old_height, old_width = preprocess_image_and_retSize(img_file, model_image_size=(img_shape, img_shape))
    st2 = time.time()
    tst_img[0, :, :, :] = img_preproc
    st3 = time.time()
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    o_p = nucleus_model.predict([tst_img , dummy_array])
    st4 = time.time()
    o_p_img = o_p[0, : , : , : , : ]
    retrieved_img_list = []
    confidence_list = []
    st3_1 = time.time()
    for j in range (o_p_img.shape[1]):
        for i in range(o_p_img.shape[0]):
            print(o_p_img[i][j][0][4])
            if o_p_img[i][j][0][4] > 0.5:
                row_c = o_p_img[i][j][0][0]
                col_c = o_p_img[i][j][0][1]
                h = o_p_img[i][j][0][2]
                w = o_p_img[i][j][0][3]
                h_actual = int(h * b_box_size)
                w_actual = int(w * b_box_size)
                row_c = row_c * b_box_size / box_scale
                col_c = col_c * b_box_size / box_scale
                row_c = row_c + (i * b_box_size)
                col_c = col_c + (j * b_box_size)
                r_top = row_c - h_actual / 2
                c_top = col_c - w_actual / 2
                r_top = max(int(r_top) , 0)
                c_top = max (int(c_top) , 0)
                r_bottom = min(int(r_top + h_actual), img_shape - 1)
                c_bottom = min(int(c_top + w_actual), img_shape - 1)
                retrieved_img_list.append(reshape([r_top, c_top, r_bottom, c_bottom], old_width, old_height, img_shape, img_shape))
                confidence_list.append(o_p_img[i][j][0][4])
    st4_1 = time.time()
    st5 = time.time()
#    retrieved_img_conv_To_img_form_list = [Image.fromarray(retrieved_img) for retrieved_img in  retrieved_img_list]
#    orig_image_list = [retrieved_img_conv_To_img_form.resize((old_width, old_height)) for retrieved_img_conv_To_img_form in retrieved_img_conv_To_img_form_list]
#    orig_image_arr_list = [np.array(orig_image, dtype='float32') for orig_image in orig_image_list]
    st6 = time.time()
    st7 = time.time()
    
    reshaped_box_coordinates = convertToNumpy(retrieved_img_list)
    
    # rem_box_list = remove_intersecting_boxes_with_non_max_supression(orig_image_arr_list, confidence_list)
    rem_box_list = remove_intersecting_boxes_with_non_max_supression(reshaped_box_coordinates, confidence_list)
    st8 = time.time()
    list_of_rle_values = []
    st9 = time.time()
    for rem_box_index in range(rem_box_list.shape[0]):
        list_of_rle_values.append(convert_to_rle_after_non_max(rem_box_list[rem_box_index], old_height, old_width))
        
#    for rem_box in rem_box_list:
#        list_of_rle_values.append(convert_to_rle_after_non_max(rem_box, orig_image_rows, orig_image_cols))
    st10 = time.time()
    rle_op[img_file_name[:-4]] = list_of_rle_values
    print("Time for preproc " + str(st2 - st1))
    print("Time for predict " + str(st4 - st3))
    print("Time for resultIteration " + str(st4_1 - st3_1))
    print("Time for unnecessary " + str(st6 - st5))
    print("Time for removing intersect boxes" + str(st8 - st7))
    print("Time for rle" + str(st10 - st9))
    print("Finished with " + img_file_name)

st11 = time.time()
checkForDuplicates(rle_op)
write_to_csv(rle_op)
st12 = time.time()
    
print("Time wrtie to csv" + str(st12 - st11))
