import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

rootdir = 'D:\Kaggle\stage1_train'

img_shape = 608

grid_size = 19

b_box_size = (img_shape/grid_size)

test_samp = np.zeros((grid_size,grid_size,5))

trn_imgs = []

masks = []

#for subdirs in os.walk(rootdir):
#    path = rootdir + subdirs[1]
#    print(path)
    
        
for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\'+ subdir
    #print(rootdir_2)
    img_dir =  os.listdir(rootdir_2)[0]
    mask_dir = os.listdir(rootdir_2)[1]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    img_preproc = preprocess_image(img_file,model_image_size = (img_shape, img_shape))
    #trn_imgs.append
    mask_files = []
    for mf in os.listdir(rootdir_2 + '\\' + mask_dir):
        mask_files.append(rootdir_2 + '\\' + mask_dir + '\\' + mf)
    for mask in mask_files:
        image_data = preprocess_mask(mask,model_image_size = (img_shape, img_shape))
        x,y,x_c,y_c,h,w = get_bounding_boxes(image_data , b_box_size)
        test_samp[x,y,:] = (1,x_c,y_c,h,w)
    
    
def preprocess_image(img_path, model_image_size):
    #image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data =image_data[:,:,:,0:3]
    return image, image_data

def preprocess_mask(mask,model_image_size):
    im = Image.open(mask, 'r')
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    return image_data

    
def get_bounding_boxes(image_data , box_size):
    x_c = 0
    y_c = 0
    tot = 0
    x_min = (image_data.shape[0] + 1)
    x_max = -1
    y_min = (image_data.shape[0] + 1)
    y_max = -1
    for i in range (image_data.shape[0]):
        for j in range(image_data.shape[1]):
            if(image_data[i][j] == 1):
                x_c = x_c + j
                y_c = y_c + i
                tot = tot + 1
                if j < x_min:
                    x_min = j
                if j > x_max:
                    x_max = j
                if i < y_min:
                    y_min = i
                if i > y_max:
                    y_max = i
    x_c = x_c / tot
    y_c = y_c / tot
    h = y_max - y_min
    w = x_max - x_min
    x = x_c // box_size
    y = y_c // box_size
    return (int(x),int(y),x_c,y_c,h,w)
    
    
    
    
        
        
    

            
