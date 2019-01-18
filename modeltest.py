
import model
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from polyps import data_transformation
from platform import system as getSystem
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import random


def load_transform_pictures(folder):
    x_train = []

    for filename in glob.glob(folder):
        im = data_transformation.load_image(filename)
        x_train.append(im)

    return(x_train)

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


def find_class(c):
    return c.argmax()

def pixel_class(c):
    if c == 1:
        return(255, 0, 0)
    elif c == 2:
        return(0, 255, 0)
    elif c == 3:
        return(0, 0, 255)
    else:
        return(0, 0, 0)


if __name__ == "__main__":


    if getSystem() == 'Windows':
        im = np.array(load_transform_pictures('polyps\\input\\data\\*.jpg'))
        test = np.array(load_transform_pictures('polyps\\test\\*.jpg'))
        output="polyps\\output\\"
    else:
        im = np.array(load_transform_pictures('polyps/input/data/*.jpg'))
        test = np.array(load_transform_pictures('polyps/test/*.jpg'))
        output = "polyps/output/"

    model = load_model('model-polyp.h5')
    lab_pred = model.predict(test, verbose=1)

    write_image(merge(lab_pred),output)