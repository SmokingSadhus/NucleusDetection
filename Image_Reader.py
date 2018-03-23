
from PIL import Image, ImageDraw, ImageFont
import numpy as np
#im = Image.open('D:\\Kaggle\\stage1_train\\1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5\\images\\1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5.png', 'r')
path = '0c320c4d08c83f73721ef5777768a5024dbae66294fd93f49d4f2e1d9fd81aa3.png'
im = Image.open(path, 'r')


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

img_shape = 608

grid_size = 19

b_box_size = (img_shape/grid_size)

test_samp = np.zeros((grid_size,grid_size,5))

######################################################################################################################################################

trn_imgs = []

masks = []

box_size = 

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
    x = x_c // 32
    y = y_c // 32
    return (int(x),int(y),x_c,y_c,h,w)
    
            
    
    





