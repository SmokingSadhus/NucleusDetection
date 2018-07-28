import csv
import os

from PIL import Image
from keras.layers import Activation
from keras.models import model_from_json
from keras.utils.generic_utils import get_custom_objects

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

rootdir = 'D:\Kaggle\stage2_test_final'

img_shape = 608

grid_size = 19

b_box_size = (img_shape / grid_size)

# b_box_size converted to box_scale for getting lower value coordinates for centers of objects row_c, col_c 
box_scale = 1

no_of_imgs = len(os.listdir(rootdir))

tst_img = np.zeros((1, img_shape, img_shape, 3))


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

def remove_intersecting_boxes_with_non_max_supression(orig_image_arr_list, confidence_list):
    boxes = np.zeros((len(orig_image_arr_list), 4))
    ct = 0
    for orig_image in orig_image_arr_list:
        row_first = -1
        col_first = -1
        row_last = -1
        col_last = -1
        for i in range (orig_image.shape[0]):
            for j in range(orig_image.shape[1]):
                if orig_image[i][j][0] == 255:
                    row_last = i
                    col_last = j
                    if row_first == -1:
                        row_first = i
                        col_first = j
        boxes[ct] = (col_first, row_first, col_last, row_last)
        ct = ct + 1 
    return non_max_suppression(boxes, confidence_list)



def non_max_suppression(boxes, confidence_list):
    tuple_list = []
    for i in range(len(boxes)):
        tuple_list.append((i, boxes[i], confidence_list[i]))
    tuple_list.sort(key=lambda tup: tup[2])
    mark_for_delete = [1] * len(tuple_list)
    
    for i in range(len(tuple_list) - 1):
        if(mark_for_delete[tuple_list[i][0]] != 0 ):
            for j in range(i + 1, len(tuple_list)):
                if(mark_for_delete[tuple_list[j][0]] != 0):
                    if(intersects(tuple_list[i][1], tuple_list[j][1])):
                        mark_for_delete[tuple_list[j][0]] = 0
    delete_indices = [i for i, e in enumerate(mark_for_delete) if e == 0]
    return np.delete(boxes, delete_indices, axis=0)
    
                    
def intersects(box1, box2):
    x_top_left = max(box1[0] , box2[0])
    x_bottom_right = min(box1[2] , box2[2])
    y_top_left = max(box1[1] , box2[1])
    y_bottom_right = min(box1[3] , box2[3])
    if ((x_top_left >=  x_bottom_right) or (y_top_left >= y_bottom_right)):
        return False
    else:
        return True 

def convert_to_rle(rem_box):
    ct = 0
    rle = []
    for j in range (rem_box.shape[1]):
        i = 0
        while i < rem_box.shape[0]:
            ct = 0
            while i < rem_box.shape[0] and rem_box[i][j][0] == 255:
                ct = ct + 1
                i = i + 1
            if ct > 0:
                i_ori = i - ct
                pixel = (j * rem_box.shape[0]) + i_ori + 1
                rle.append(pixel)
                rle.append(ct)
            i = i + 1
    return rle


def convert_to_rle_after_non_max(rem_box, rows, cols):
    row_first = int(rem_box[0])
    col_first = int(rem_box[1])
    row_last = int(rem_box[2])
    col_last = int(rem_box[3])
    height_of_box = row_last - row_first + 1
    rle = []
    for j in range(col_first, col_last + 1):
        pixel_st = (j * rows) + row_first + 1
        rle.append(pixel_st)
        rle.append(height_of_box)


def write_to_csv(rle_op):
    with open('output.csv', 'w') as csvfile:
        fieldnames = ['ImageId', 'EncodedPixels']
        writer = csv.DictWriter(csvfile, lineterminator='\n' , fieldnames=fieldnames)
        writer.writeheader()
        for image_id in rle_op.keys():
            list_of_masks = rle_op[image_id]
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


get_custom_objects().update({'custom_activation': Activation(custom_activation)})

###### Code starts from here #######################################

json_file = open('nucleus_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
nucleus_model_loaded = model_from_json(loaded_model_json)
# load weights into new model
nucleus_model_loaded.load_weights("nucleus_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
nucleus_model_loaded.compile(loss='mean_squared_error', optimizer='adam')

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
    o_p = nucleus_model.predict(tst_img)
    st4 = time.time()
    o_p_img = o_p[0, :, :, :]
    retrieved_img_list = []
    confidence_list = []
    st3_1 = time.time()
    for j in range (o_p_img.shape[1]):
        for i in range(o_p_img.shape[0]):
            if o_p_img[i][j][0] > 0.5:
                row_c = o_p_img[i][j][1]
                col_c = o_p_img[i][j][2]
                h = o_p_img[i][j][3]
                w = o_p_img[i][j][4]
                h_actual = int(h * b_box_size)
                w_actual = int(w * b_box_size)
                row_c = row_c * b_box_size / box_scale
                col_c = col_c * b_box_size / box_scale
                row_c = row_c + (i * b_box_size)
                col_c = col_c + (j * b_box_size)
                r_top = row_c - h_actual / 2
                c_top = col_c - w_actual / 2
                r_top = int(r_top)
                c_top = int(c_top)
                retrieved_img = np.zeros((img_shape, img_shape, 3), dtype=np.uint8)
                for r in range(r_top, r_top + h_actual + 1):
                    for c in range(c_top, c_top + w_actual + 1):
                        retrieved_img[r][c][0] = 255
                        retrieved_img[r][c][1] = 255
                        retrieved_img[r][c][2] = 255
                retrieved_img_list.append(retrieved_img)
                confidence_list.append(o_p_img[i][j][0])
    st4_1 = time.time()
    st5 = time.time()
    retrieved_img_conv_To_img_form_list = [Image.fromarray(retrieved_img) for retrieved_img in  retrieved_img_list]
    orig_image_list = [retrieved_img_conv_To_img_form.resize((old_width, old_height)) for retrieved_img_conv_To_img_form in retrieved_img_conv_To_img_form_list]
    orig_image_arr_list = [np.array(orig_image, dtype='float32') for orig_image in orig_image_list]
    st6 = time.time()
    st7 = time.time()
    
    # orig_image_rows and orig_image_cols should ideally be equal to old_height and old_width
    if len(orig_image_arr_list) == 0:
        orig_image_rows = old_height
        orig_image_cols = old_width
    else:
        orig_image_rows = (orig_image_arr_list[0]).shape[0]
        orig_image_cols = (orig_image_arr_list[0]).shape[1]
    
    rem_box_list = remove_intersecting_boxes_with_non_max_supression(orig_image_arr_list, confidence_list)
    
    st8 = time.time()
    list_of_rle_values = []
    st9 = time.time()
    for rem_box_index in range(rem_box_list.shape[0]):
        list_of_rle_values.append(convert_to_rle_after_non_max(rem_box_list[rem_box_index], orig_image_rows, orig_image_cols))
        
#    for rem_box in rem_box_list:
#        list_of_rle_values.append(convert_to_rle_after_non_max(rem_box, orig_image_rows, orig_image_cols))
    st10 = time.time()
    rle_op[img_file_name] = list_of_rle_values
    print("Time for preproc " + str(st2 - st1))
    print("Time for predict " + str(st4 - st3))
    print("Time for resultIteration " + str(st4_1 - st3_1))
    print("Time for unnecessary " + str(st6 - st5))
    print("Time for removing intersect boxes" + str(st8 - st7))
    print("Time for rle" + str(st10 - st9))
    print("Finished with " + img_file_name)

st11 = time.time()
write_to_csv(rle_op)
st12 = time.time()
    
print("Time wrtie to csv" + str(st12 - st11))
