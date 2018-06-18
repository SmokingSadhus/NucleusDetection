import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split
import csv
#################

#image = Image.open("D:\Kaggle\extendedTestCrop.png")
#print(image.size)
#image_data = np.array(image, dtype='float32')
#print(image_data.shape)

#exit()
#############################

rootdir = 'D:\Kaggle\stage1_test'

img_shape = 608

grid_size = 19

b_box_size = (img_shape/grid_size)

#b_box_size converted to box_scale for getting lower value coordinates for centers of objects row_c, col_c 
box_scale = 1

no_of_imgs = len(os.listdir(rootdir))

tst_img = np.zeros((1,img_shape,img_shape,3))


def preprocess_image(img_path, model_image_size):
    #image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    w,h = image.size
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    #resized_image = image.resize(model_image_size)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    #image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data =image_data[:,:,0:3]
    return image, image_data,h,w


# Not a good method for eliminating boxes, as we remove one of the 2 intersecting boxes(box with lower confidence). Use method in coursera deep learning course instead.
def remove_intersecting_boxes(orig_image_arr_list,confidence_list):
    pixel_to_img = dict()
    int count = 0
    list_of_removables = []
    for orig_image in orig_image_arr_list:
        for i in range (orig_image.shape[0]):
            for j in range(orig_image.shape[1]):
                if orig_image[i][j][0] == 255:
                    key = ((j*orig_image.shape[0]) + i)
                    if key in pixel_to_img:
                        old = pixel_to_img[key]
                        if confidence_list[old] >= confidence_list[count]:
                            list_of_removables.add(count)
                            #Need to fond a way to break here
                        else:
                            list_of_removables.add(old)
                            pixel_to_img[key] = count
                    else:
                        pixel_to_img[key] = count
        count = count + 1
    ret_list = []
    for idx, val in enumerate(orig_image_arr_list):
        if idx not in list_of_removables:
            ret_list.add(val)
    return ret_list      

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
                rle.add(pixel)
                rle.add(ct)
    return rle

def write_to_csv(rle_op):
    with open('output.csv', 'w') as csvfile:
    fieldnames = ['ImageId', 'EncodedPixels']
    writer = csv.DictWriter(csvfile, lineterminator='\n' ,fieldnames=fieldnames)
    writer.writeheader()
    for image_id in rle_op.keys():
        list_of_masks = rle_op[image_id]
        for mask in list_of_masks:
            mask_str = ''
            for pix_dist in mask:
                mask_str = mask_str + pix_dist + ' '
            writer.writerow({'ImageId':image_id,EncodedPixels:mask_str})

rle_op = dict()

for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\'+ subdir
    #print(img_count)
    img_dir =  os.listdir(rootdir_2)[0]
    mask_dir = os.listdir(rootdir_2)[1]
    img_file_name = os.listdir(rootdir_2 + '\\' + img_dir)[0]
    #img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + img_file_name
    #print(img_file)
    img, img_preproc,old_height,old_width = preprocess_image_and_retSize(img_file,model_image_size = (img_shape, img_shape))
    tst_img[1,:,:,:] =  img_preproc
    o_p = nucleus_model.predict(tst_img)
    o_p_img = o_p[1,:,:,:]
    retrieved_img_list = []
    confidence_list = []
    #changed iterating order to match submission format
    for j in range (o_p_img.shape[1]):
        for i in range(o_p_img.shape[0]):
            if o_p_img[i][j][0]  > 0.5:
                row_c = o_p_img[i][j][1]
                col_c = o_p_img[i][j][2]
                h = o_p_img[i][j][3]
                w = o_p_img[i][j][4]
                h_actual = h * b_box_size
                w_actual = w * b_box_size
                row_c = row_c * b_box_size/box_scale
                col_c = col_c * b_box_size/box_scale
                row_c = row_c + (i * b_box_size)
                col_c = col_c + (j * b_box_size)
                r_top = row_c - h_actual/2
                c_top = col_c - w_actual/2
                r_top = int(r_top)
                c_top = int(c_top)
                retrieved_img = np.zeros((img_shape,img_shape,3))
                for r in range(r_top,r_top+h_actual+1):
                    for c in range(c_top,c_top + w_actual+1):
                        retrieved_img[r][c][0] = 255
                        retrieved_img[r][c][1] = 255
                        retrieved_img[r][c][2] = 255
                retrieved_img_list.add(retrieved_img)
                confidence_list.add(o_p_img[i][j][0])
    retrieved_img_conv_To_img_form_list = [ Image.fromarray(retrieved_img) for retrieved_img in  retrieved_img_list]
    orig_image_list = [retrieved_img_conv_To_img_form.resize((old_width,old_height)) for retrieved_img_conv_To_img_form in retrieved_img_conv_To_img_form_list]
    orig_image_arr_list = [np.array(orig_image, dtype='float32') for orig_image in orig_image_list]
    rem_box_list = remove_intersecting_boxes(orig_image_arr_list,confidence_list)
    list_of_rle_values = []
    for rem_box in rem_box_list:
        list_of_rle_values.add(convert_to_rle(rem_box))
    rle_op[img_file_name] =  list_of_rle_values       
        
        
        
    
    
