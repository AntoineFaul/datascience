from keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import backend as K
import glob

from data import augmentation, binary_masks, manager
from config import config, model_evaluate
from .model import u_net
import .contingency


def pixel_class(c):
    return config['color']['rgb'][c]
        
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

    return final_images

def write_image(array, output_path):
    manager.clean_folder(output_path)

    names = manager.list_dir(manager.make_path('polyps', 'test', 'data'))

    index = 0
    img_store = []

    for image in array:
        img = Image.new('RGB', config['image_size'], 'white')

        for i in range(config['image_max']):
            for j in range(config['image_max']):
                img.putpixel((i, j), image[i][j])

        n = (names[index].split(config['path_sep']))[-1]
        name = '{}'.format(n)
        img.save(manager.make_path(output_path, name))
        img_store.append(np.array(img))

        index += 1

    return img_store

def result_jaccard_coeff(img1, img2):
    img1_t = K.variable(img1)
    img2_t = K.variable(img2)

    return K.eval(jacard_coef(img1_t, img2_t))

def model_predict(model, img_test, output_path):
    manager.clean_folder(output_path)
    
    lab_pred = model.predict(img_test, verbose = 1)
    write_image(merge(lab_pred), output_path)

    contingency.overall_table(lab_pred, mask_test)
    contingency.table(lab_pred, mask_test)


def execute(run_data_augmentation = True):
    if run_data_augmentation:
        augmentation.execute()

    model = u_net()
    model.compile(optimizer = Adam(lr = 1e-4), loss = config['classification']['loss'] , metrics = config['classification']['metrics'])

    img_train, img_val, img_test = manager.load_imgs()

    mask_train = np.array(data_transformation.create_binary_masks(path = manager.make_path('polyps', 'training', 'label')), dtype = "float32") 
    mask_val = np.array(data_transformation.create_binary_masks(path = manager.make_path('polyps', 'validation', 'label')), dtype = "float32") 
    mask_test = np.array(data_transformation.create_binary_masks(path = manager.make_path('polyps', 'test', 'label')), dtype = "float32")

    output_path = manager.make_path('polyps', 'output', 'classification')
  
    model.fit(x = img_train,
                y = mask_train,
                validation_data = (img_val, mask_val),
                epochs = config['fit']['epochs'],
                shuffle = config['fit']['shuffle'],
                batch_size = config['fit']['batch_size'],
                callbacks = config['fit']['callbacks']  
            )

    model_predict(model, img_test, output_path)
    model_evaluate(model, img_test, mask_test)
    
