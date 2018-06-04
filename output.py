import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split
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

for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\'+ subdir
    #print(img_count)
    img_dir =  os.listdir(rootdir_2)[0]
    mask_dir = os.listdir(rootdir_2)[1]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    #print(img_file)
    img, img_preproc,old_height,old_width = preprocess_image_and_retSize(img_file,model_image_size = (img_shape, img_shape))
    tst_img[1,:,:,:] =  img_preproc
    o_p = nucleus_model.predict(tst_img)
    o_p_img = o_p[1,:,:,:]
    retrieved_img = np.zeros((img_shape,img_shape,3))
    for i in range (o_p_img.shape[0]):
        for j in range(o_p_img.shape[1]):
            if o_p_img[i][j][0]  > 0.5:
                row_c = o_p_img[i][j][1]
                col_c = o_p_img[i][j][2]
                h = o_p_img[i][j][3]
                w = o_p_img[i][j][4]
                h_actual = h * box_size
                w_actual = w * box_size
                row_c = row_c * box_size/box_scale
                col_c = col_c * box_size/box_scale
                row_c = row_c + (i * b_box_size)
                col_c = col_c + (j * b_box_size)
                r_top = row_c - h_actual/2
                c_top = col_c - w_actual/2
                r_top = int(r_top)
                c_top = int(c_top)
                for r in range(r_top,r_top+h_actual+1):
                    for c in range(c_top,c_top + w_actual+1):
                        retrieved_img[r][c][0] = 255
                        retrieved_img[r][c][1] = 255
                        retrieved_img[r][c][2] = 255
    retrieved_img_conv_To_img_form = Image.fromarray(retrieved_img)
    orig_image = retrieved_img_conv_To_img_form.resize((old_width,old_height))
    orig_image_arr = np.array(orig_image, dtype='float32')
    
      
            

    #trn_imgs.append
    mask_files = []
    for mf in os.listdir(rootdir_2 + '\\' + mask_dir):
        mask_files.append(rootdir_2 + '\\' + mask_dir + '\\' + mf)
    for mask in mask_files:
        #print(mask)
        image_data = preprocess_mask(mask,model_image_size = (img_shape, img_shape))
        row,col,row_c,col_c,h,w = get_bounding_boxes(image_data , b_box_size)
        mask_op[img_count, row , col ,:] = (1,row_c,col_c,h,w)
    img_count = img_count + 1
