
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
#im = Image.open('D:\\Kaggle\\stage1_train\\1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5\\images\\1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5.png', 'r')

rootdir = 'D:\Kaggle\stage1_train'

im_size = set()

for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\'+ subdir
    #print(img_count)
    img_dir =  os.listdir(rootdir_2)[0]
    mask_dir = os.listdir(rootdir_2)[1]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    im = Image.open(img_file)
    image_data = np.array(im, dtype='float32')
    im_size.add(image_data.shape)

exit()

path = 'D:\\Kaggle\\stage1_train\\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552\\masks\\5522143fa8723b66b1e0b25331047e6ae6eeec664f7c8abeba687e0de0f9060a.png'

path = 'D:\\Kaggle\\stage1_train\\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552\\masks\\07a9bf1d7594af2763c86e93f05d22c4d5181353c6d3ab30a345b908ffe5aadc.png'



def preprocess_mask(mask,model_image_size):
    im = Image.open(mask, 'r')
    #im = im.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    im = im.resize(model_image_size)
    image_data = np.array(im, dtype='float32')
    return image_data

def preprocess_image(img_path, model_image_size):
    #image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    #resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    resized_image = image.resize(model_image_size)
    image_data = np.array(resized_image, dtype='float32')
    print(image_data.shape)
    image_data /= 255.
    #image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data =image_data[:,:,0:3]
    return image, image_data

#image_data = preprocess_mask(path,model_image_size =(608,608))

img_path = 'D:\\Kaggle\\stage1_train\\0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e\\images\\0b0d577159f0d6c266f360f7b8dfde46e16fa665138bf577ec3c6f9c70c0cd1e.png'

img, image_data = preprocess_image(img_path, model_image_size =(608,608))

print(image_data.shape[0])
print(image_data.shape[1])

exit()

#im = Image.open(path)
#image_data = np.array(im)

#print(image_data)

st = set()

for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        if(int(image_data[i][j]) != 0):
            st.add(image_data[i][j])

for i in st:
    print(i)

exit()

image_data = np.array(im, dtype='float32')
#image_data =image_data[:,:,0:3]
#print(image_data.shape)
#print(image_data)

def preprocess_image(img_path, model_image_size):
    #image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


image, image_data = preprocess_image(path, model_image_size = (608, 608))

#image_data =image_data[:,:,:,0:3]

print(image_data)

print(image_data.shape)

exit()

pix_val = list(im.getdata())
print(pix_val[0])
print(len(pix_val))

print(im.getdata()[4156])

####(1, 608, 608, 4)

###############################################################

rootdir = 'D:\Kaggle\stage1_train'

for dirs in os.walk(rootdir):
    print(dirs) 

img_shape = 608

grid_size = 19

b_box_size = (img_shape/grid_size)

test_samp = np.zeros((grid_size,grid_size,5))

trn_imgs = []

masks = []

#box_size = 

for mask in masks:
    im = Image.open(mask, 'r')
    resized_image = image.resize(tuple(reversed((img_shape,img_shape))), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    x,y,x_c,y_c,h,w = get_bounding_boxes(image_data , box_size)
    test_samp[x,y,:] = (1,x_c,y_c,h,w)
    
    
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
    
###################################################


    






