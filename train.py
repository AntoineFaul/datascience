import model
from polyps import data_transformation
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import sys

import random


def load_transform_pictures(folder):
    maxsize = (224,224)
    x_train=[]
    for filename in glob.glob(folder):
        im=Image.open(filename)
        im.thumbnail(maxsize) # find other technique
        im = np.array(im,dtype="float32")
        im = im /255 # for learing faster
        x_train.append(im)
    return(x_train)


def pixel_class(c):
    if c == 0:
        return(0,0,0)
    if c == 1:
        return(255,0,0)
    if c == 2:
        return(0,255,0)
    if c == 3:
        return(0,0,255)
        
def find_class(c):
    return c.argmax()

def merge(array):
    final_images = []
    for image in array:
        cimage = []
        for row in image:
            crow = []
            for pixel in row:
                crow.append(pixel_class(find_class(pixel)))
            cimage.append(crow)
        final_images.append(cimage)
    return(final_images)
    
def write_image(array, directory):
    index = 0
    for image in array:
        index = index +1
        img = Image.new("RGB", (224,224), "white")
        for i in range(224):
            for j in range(224):
                
                img.putpixel((i,j),image[i][j])
        name = '{0:04}'.format(index) + "_output.jpg"
        img.save(directory+name)

if __name__ == "__main__":
    batch_size = 64
    model = model.u_net() #what does the Adam optimizer do
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy' , metrics = ['accuracy'])#,pixel_accuracy])
    if sys.platform == 'linux':
        im = np.array(load_transform_pictures('polyps/data/*.jpg'))
        test = np.array(load_transform_pictures('polyps/test/*.jpg'))
        output = "polyps/output/"
    else:
        im = np.array(load_transform_pictures('polyps\\data\\*.jpg'))
        test = np.array(load_transform_pictures('polyps\\data\\*.jpg'))
        output="polyps\\output\\"
    mask = np.array(data_transformation.create_binary_masks()) 
    model.fit(x = im,y=mask,
                        steps_per_epoch = 1048//batch_size,#1048//batch_size,
                        validation_split = 0.2,
                        validation_steps = 128//batch_size,
                        epochs = 5, 
                        )
    
        
    lab_pred = model.predict(test)
    write_image(merge(lab_pred),output)
