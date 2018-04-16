import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split

#rootdir = 'D:\Kaggle\stage1_train'

rootdir = 'D:\Kaggle\mask_and_train_check'

img_shape = 608

grid_size = 19

b_box_size = (img_shape/grid_size)

#b_box_size converted to box_scale for getting lower value coordinates for centers of objects row_c, col_c 
box_scale = 5

#test_samp = np.zeros((grid_size,grid_size,5))

no_of_imgs = len(os.listdir(rootdir))

trn_imgs = np.zeros((no_of_imgs,img_shape,img_shape,3))

mask_op = np.zeros((no_of_imgs,grid_size,grid_size,5))

trn_tst_split = 0.5

#for subdirs in os.walk(rootdir):
#    path = rootdir + subdirs[1]
#    print(path)
    
img_count = 0

def preprocess_image(img_path, model_image_size):
    #image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    #resized_image = image.resize(model_image_size)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    #image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data =image_data[:,:,0:3]
    return image, image_data

def preprocess_mask(mask,model_image_size):
    im = Image.open(mask, 'r')
    #resized_image = im.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    resized_image = im.resize(model_image_size)
    image_data = np.array(resized_image, dtype='float32')
    return image_data

    
#def get_bounding_boxes(image_data , box_size):
#    x_c = 0
#    y_c = 0
#    tot = 0
#    x_min = (image_data.shape[0] + 1)
#    x_max = -1
#    y_min = (image_data.shape[0] + 1)
#    y_max = -1
#    for i in range (image_data.shape[0]):
#        for j in range(image_data.shape[1]):
#            if(int(image_data[i][j]) == 255):
#                x_c = x_c + j
#                y_c = y_c + i
#                tot = tot + 1
#                if j < x_min:
#                    x_min = j
#                if j > x_max:
#                    x_max = j
#                if i < y_min:
#                    y_min = i
#                if i > y_max:
#                    y_max = i
    #if tot == 0:
    #    print('problem')
    #    return (0,0,0,0,0,0)
#    x_c = x_c / tot
#    y_c = y_c / tot
#    h = y_max - y_min
#    w = x_max - x_min
#    x = x_c // box_size
#    y = y_c // box_size
#    return (int(x),int(y),x_c,y_c,h,w)

def get_bounding_boxes(image_data , box_size):
    row_c = 0
    col_c = 0
    tot = 0
    row_min = (image_data.shape[0] + 1)
    row_max = -1
    col_min = (image_data.shape[1] + 1)
    col_max = -1
    for i in range (image_data.shape[0]):
        for j in range(image_data.shape[1]):
            if(int(image_data[i][j]) == 255):
                row_c = row_c + i
                col_c = col_c + j
                tot = tot + 1
                if j < col_min:
                    col_min = j
                if j > col_max:
                    col_max = j
                if i < row_min:
                    row_min = i
                if i > row_max:
                    row_max = i
    #if tot == 0:
    #    print('problem')
    #    return (0,0,0,0,0,0)
    row_c = row_c / tot
    col_c = col_c / tot
    row_c_box = (((row_c % box_size) * box_scale) / box_size)
    col_c_box = (((col_c % box_size) * box_scale) / box_size)
    h = row_max - row_min
    #Height with respect to box size
    h = h/box_size
    w = col_max - col_min
    #Width with respect to box size
    w = w/box_size
    row = row_c // box_size
    col = col_c // box_size
    return (int(row),int(col),row_c_box,col_c_box,h,w)


for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\'+ subdir
    print(img_count)
    img_dir =  os.listdir(rootdir_2)[0]
    mask_dir = os.listdir(rootdir_2)[1]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    print(img_file)
    img, img_preproc = preprocess_image(img_file,model_image_size = (img_shape, img_shape))
    trn_imgs[img_count,:,:,:] =  img_preproc
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
    
#########

#np.save('trn_chk.npy', trn_imgs)
#np.save('mask_chk.npy', mask_op)

np.save('trn.npy', trn_imgs)

np.save('mask.npy', mask_op)

data_train, data_test, labels_train, labels_test = train_test_split(trn_imgs, mask_op, test_size=trn_tst_split, random_state=42)

np.save('data_train.npy',data_train)
np.save('data_test.npy',data_test)
np.save('labels_train.npy',labels_train)
np.save('labels_test.npy',labels_test)

#exit()

#d = np.load('test3.npy')

#f_img = mask_op[0]

#f_img = mask_op[1]

#f_img = labels_train[0]

##########################
#for i in range(f_img.shape[0]):
#	for j in range(f_img.shape[1]):
#		if  int(f_img[i][j][0]) != 0:
#			print (f_img[i,j,:])
#			print (str(i) + '  ' + str(j))
 ################################   
    
        
        
    

            
