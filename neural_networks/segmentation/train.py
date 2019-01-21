import cv2
import numpy as np
from matplotlib.pyplot import imsave
from keras.optimizers import Adam

from .model import u_net
from .save import write_images
from data import augmentation, manager
from config import config, model_evaluate, model_predict


def load_masks():
    mask_train = np.array(manager.load_images(manager.make_path('polyps_pixel', 'training', 'label')), dtype = config['dtype'])
    mask_val = np.array(manager.load_images(manager.make_path('polyps_pixel', 'validation', 'label')), dtype = config['dtype'])
    mask_test = np.array(manager.load_images(manager.make_path('polyps_pixel', 'test', 'label')), dtype = config['dtype'])

    return (mask_train, mask_val, mask_test)


def execute(run_data_augmentation = True):
    if run_data_augmentation:
        augmentation.execute()

    output_path = manager.make_path('polyps_pixel', 'output', 'segmentation')
    manager.clean_folder(output_path)

    model = u_net()
    model.compile(optimizer = Adam(lr = 1e-4), loss = config['segmentation']['loss'], metrics = config['segmentation']['metrics'])

    (img_train, img_val, img_test) = manager.load_imgs()
    (mask_train, mask_val, mask_test) = load_masks()
    
    model.fit(x = img_train,
                y = mask_train,
                validation_data = (img_val, mask_val),
                batch_size = config['fit']['batch_size'],
                shuffle = config['fit']['shuffle'],
                epochs = config['fit']['epochs']
            )
    
    model_predict(model, write_images, img_test, output_path)
    model_evaluate(model, img_test, mask_test)
