from keras.models import Model, load_model
from polyps import data_augmentation, data_transformation, file_manager as fm
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import backend as K
import glob
import model
from config import config


def load_transform_pictures(folder):
    x_train = []

    for filename in glob.glob(folder):
        im = fm.load_image(filename)
        x_train.append(im)

    return(x_train)

def load_file_name(folder):
    x_train = []
    for filename in glob.glob(folder):
        x_train.append(filename)
    return(x_train)

def pixel_class(c):
    if c == 1:
        return config['color']['rgb']['red']
    elif c == 2:
        return config['color']['rgb']['green']
    elif c == 3:
        return config['color']['rgb']['blue']
    else:
        return config['color']['rgb']['black']
        
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

def write_image(array, directory, file_name):
    fm.clean_folder(fm.make_path('polyps', 'output'))
    index = 0
    img_store = []

    for image in array:
        img = Image.new('RGB', config['image_size'], "white")

        for i in range(config['image_max']):
            for j in range(config['image_max']):
                img.putpixel((i, j), image[i][j])
        n = (file_name[index].split(config['path_sep']))[-1]
        name = '{}'.format(n)
        img.save(fm.make_path(directory, name))
        img_store.append(np.array(img))
        index = index+1
    return img_store

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred): # between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


if __name__ == "__main__":
    data_augmentation.execute()

    batch_size = config['batch_size']
    model = model.u_net(IMG_SIZE = config['image_dimension']) #what does the Adam optimizer do

    model.compile(optimizer = Adam(lr = 1e-4), loss = config['loss'] , metrics = config['metrics'])


    im = np.array(load_transform_pictures(fm.make_path('polyps', 'input', 'data', '*.jpg')))
    test = np.array(load_transform_pictures(fm.make_path('polyps', 'test','data', '*.jpg')))
    test_name = np.array(load_file_name(fm.make_path('polyps', 'test', 'data', '*.jpg')))
    output = fm.make_path('polyps', 'output','data')
    path = fm.make_path('polyps', 'input', 'label')
    path_test = fm.make_path('polyps', 'test', 'label')
    mask = np.array(data_transformation.create_binary_masks(path = path)) 
    mask_test = np.array(data_transformation.create_binary_masks(path = path_test))
  
    history = model.fit(x = im, y = mask,
                        validation_split = config['validation_split'],
                        steps_per_epoch = config['fit']['steps_per_epoch'],
                        validation_steps = config['fit']['validation_steps'],
                        epochs = config['fit']['epochs'],
                        shuffle = config['fit']['shuffle'],
                        batch_size = config['fit']['batch_size'],
                        class_weight = config['fit']['class_weight'],
                        callbacks = config['fit']['callbacks']  
                    )

    history = history.history
    lab_pred = model.predict(test, verbose = 1)
    evaluate = model.evaluate(x = test, y = mask_test, batch_size = batch_size)
    display_im = write_image(merge(lab_pred), output, test_name)
    plt.imshow(display_im[0])#plots the first picture
    print("Evaluation : Loss: "+ str(evaluate[0]) + ", Accuracy: " + str(evaluate[1]) + ", Dice Coefficient: " + str(evaluate[2]) + ", Jacard Coefficient: " + str(evaluate[3]))
