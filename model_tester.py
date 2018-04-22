import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split

json_file = open('nucleus_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
nucleus_model_loaded = model_from_json(loaded_model_json)
# load weights into new model
nucleus_model_loaded.load_weights("nucleus_model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
nucleus_model_loaded.compile(loss=yolo_loss, optimizer='adam')

rootdir = 'D:\Kaggle\mask_and_train_check'

img_shape = 608

trn_imgs = np.zeros((1,img_shape,img_shape,3))

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

for subdir in os.listdir(rootdir):
    rootdir_2 = rootdir + '\\'+ subdir
    #print(img_count)
    img_dir =  os.listdir(rootdir_2)[0]
    mask_dir = os.listdir(rootdir_2)[1]
    img_file = rootdir_2 + '\\' + img_dir + '\\' + os.listdir(rootdir_2 + '\\' + img_dir)[0]
    #print(img_file)
    img, img_preproc = preprocess_image(img_file,model_image_size = (img_shape, img_shape))
    trn_imgs[img_count,:,:,:] =  img_preproc
    #trn_imgs.append
    img_count = img_count + 1




o_p = nucleus_model_loaded.predict(trn_imgs, batch_size=1)

f_img = o_p[0]

for i in range(f_img.shape[0]):
	for j in range(f_img.shape[1]):
		if  int(f_img[i][j][0]) != 0:
			print (f_img[i,j,:])
			print (str(i) + '  ' + str(j))






