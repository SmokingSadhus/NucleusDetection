
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

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
