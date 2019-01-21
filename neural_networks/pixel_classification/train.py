from keras.optimizers import Adam
import numpy as np

from data import augmentation, binary_masks, manager
from config import config, model_evaluate, model_predict
from .save import write_images
from . import contingency
from .model import u_net


<<<<<<< HEAD
def load_masks(): #load in the mask data
    mask_train = np.array(binary_masks.create(manager.make_path('polyps', 'training', 'label')), dtype = config['dtype']) 
    mask_val = np.array(binary_masks.create(manager.make_path('polyps', 'validation', 'label')), dtype = config['dtype']) 
    mask_test = np.array(binary_masks.create(manager.make_path('polyps', 'test', 'label')), dtype = config['dtype'])
=======
def load_masks():
    mask_train = np.array(binary_masks.create(manager.make_path('polyps_pixel', 'training', 'label')), dtype = config['dtype']) 
    mask_val = np.array(binary_masks.create(manager.make_path('polyps_pixel', 'validation', 'label')), dtype = config['dtype']) 
    mask_test = np.array(binary_masks.create(manager.make_path('polyps_pixel', 'test', 'label')), dtype = config['dtype'])
>>>>>>> 5298e682b06d76acd6bfd042c18ae27190a16ceb

    return (mask_train, mask_val, mask_test)

def model_predict_extra(model, img_test, mask_test, output_path): #predicts testing images and display a confusion matrix for each class and overall
    lab_pred = model_predict(model, write_images, img_test, output_path)

    contingency.overall_table(lab_pred, mask_test)
    contingency.table(lab_pred, mask_test)


def execute(run_data_augmentation = True): #main function for running the pixel-wise classification
    if run_data_augmentation:
        augmentation.execute()

    output_path = manager.make_path('polyps_pixel', 'output', 'classification')
    manager.clean_folder(output_path)

    model = u_net()
    model.compile(optimizer = Adam(lr = 1e-4), loss = config['classification']['loss'] , metrics = config['classification']['metrics'])

    img_train, img_val, img_test = manager.load_imgs()
    mask_train, mask_val, mask_test = load_masks()
  
    model.fit(x = img_train, 
                y = mask_train,
                validation_data = (img_val, mask_val),
                epochs = config['fit']['epochs'],
                shuffle = config['fit']['shuffle'],
                batch_size = config['fit']['batch_size'],
                callbacks = config['fit']['callbacks']  
            )

    model_predict_extra(model, img_test, mask_test, output_path)
    model_evaluate(model, img_test, mask_test)
