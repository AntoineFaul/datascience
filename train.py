from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from polyps import data_transformation
from file_manager import make_path, load_image
from platform import system as getSystem
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import random
import model


def load_transform_pictures(folder):
    x_train = []

    for filename in glob.glob(folder):
        im = load_image(filename)
        x_train.append(im)

    return(x_train)

def pixel_class(c):
    if c == 1:
        return(255, 0, 0)
    elif c == 2:
        return(0, 255, 0)
    elif c == 3:
        return(0, 0, 255)
    else:
        return(0, 0, 0)
        
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
    batch_size = 50
    model = model.u_net() #what does the Adam optimizer do

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy' , metrics = ['accuracy'])#, pixel_accuracy])

    im = np.array(load_transform_pictures(make_path('polyps', 'input', 'data', '*.jpg')))
    test = np.array(load_transform_pictures(make_path('polyps', 'test', '*.jpg')))
    output = make_path('polyps', 'output')

    mask = np.array(data_transformation.create_binary_masks()) 
    model.fit(x = im,
                y = mask,
                steps_per_epoch = 1048//batch_size,
                validation_split = 0.2,
                validation_steps = 128//batch_size,
                epochs = 1
            )
    
    model = load_model('model-polyp.h5')
    lab_pred = model.predict(test, verbose=1)

    write_image(merge(lab_pred),output)
