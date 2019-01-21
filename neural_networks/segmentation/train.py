import cv2
import numpy as np
from keras.optimizers import Adam

from .model import u_net
from data import augmentation, manager
from config import config, jacard_coef_loss, jacard_coef, dice_coef


def model_evaluate(model, img_test, mask_test):
    evaluate = model.evaluate(x = img_test, y = mask_test, batch_size = config['batch_size'])
    print("\nEvaluation : Loss: "+ str(round(evaluate[0], 4)) + ", Accuracy: " + str(round(evaluate[1], 4)) + ", Dice Coefficient: " + str(round(evaluate[2], 4)) + ", Jacard Coefficient: " + str(round(evaluate[3], 4)))

def model_predict(model, img_test, output_path):
    manager.clean_folder(output_path)
    
    lab_pred = model.predict(img_test, verbose = 1)
    names = manager.list_dir(manager.make_path('polyps', 'test', 'data'))

    for i in range(len(lab_pred)):
        cv2.imwrite(manager.make_path(output_path, names[i]), lab_pred[i])


def execute(run_data_augmentation = True):
    if run_data_augmentation:
        augmentation.execute()

    model = u_net()
    model.compile(optimizer = Adam(lr = 1e-4), loss = config['segmentation']['loss'], metrics = config['segmentation']['metrics'])

    (img_train, img_val, img_test) = manager.load_imgs()
    (mask_train, mask_val, mask_test) = manager.load_masks()

    output_path = manager.make_path('polyps', 'output', 'segmentation')
    
    model.fit(x = img_train,
                y = mask_train,
                validation_data = (img_val, mask_val),
                batch_size = config['fit']['batch_size'],
                shuffle = config['fit']['shuffle'],
                epochs = config['fit']['epochs']
            )
    
    model_predict(model, img_test, output_path)
    model_evaluate(model, img_test, mask_test)
