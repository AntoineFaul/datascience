from keras.callbacks import EarlyStopping, ModelCheckpoint
from platform import system as getSystem
import keras.backend as K
import numpy as np


PATH_SEP = '\\' if getSystem() == 'Windows' else '/'

CHANNELS = 3
IMG_MAX_SIZE = 224
BATCH_SIZE = 8


def model_predict(model, img_save, img_test, output_path):
    lab_pred = model.predict(img_test, verbose = 1)
    img_save(lab_pred, output_path)

    return lab_pred

def model_evaluate(model, img_test, mask_test):
    evaluate = model.evaluate(x = img_test, y = mask_test, batch_size = config['batch_size'])
    print("\nEvaluation : Loss: "+ str(round(evaluate[0], 4)) + ", Accuracy: " + str(round(evaluate[1], 4)) + ", Dice Coefficient: " + str(round(evaluate[2], 4)) + ", Jacard Coefficient: " + str(round(evaluate[3], 4)))


def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        y_pred = K.clip(y_pred, K.epsilon(), 1- K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)

        return loss

    return loss

def jacard_coef(y_true, y_pred, smooth = 100.0): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    y_inter = K.sum(y_true_f * y_pred_f)
    y_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (y_inter + smooth) / (y_sum - y_inter + smooth)

def jacard_coef_loss(y_true, y_pred, smooth = 100.0):
    return (1 - jacard_coef(y_true, y_pred)) * smooth

def result_jaccard_coeff(img1, img2):
    img1_t = K.variable(img1)
    img2_t = K.variable(img2)

    return K.eval(jacard_coef(img1_t, img2_t))

def dice_coef(y_true, y_pred, smooth = 100.0): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_inter = K.sum(y_true_f * y_pred_f)
    y_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2.0 * y_inter + smooth) / (y_sum + smooth)

def dice_coef_loss(y_true, y_pred, smooth = 100.0):
    return (1 - dice_coef(y_true, y_pred)) * smooth

earlystopper = EarlyStopping(monitor = 'val_loss', #stop when validation loss decreases
								min_delta = 0.2, #if val_loss < 0 it stops
								patience = 10, #minimum amount of epochs
								verbose = 1,
								mode = 'auto') # print a text

checkpointer = ModelCheckpoint('neural_networks' + PATH_SEP + 'pixel_classification' + PATH_SEP + 'model.h5', verbose = 1, save_best_only = True, monitor ='val_jacard_coef', mode = 'max')


config = {
	'path_sep': PATH_SEP,
	'except_files': ['.gitkeep'],

	'dtype': 'float32',

	'neurons': 64,

	'validation_split': 0.2,
	'test_split': 0.2,

	'image_max': IMG_MAX_SIZE,
	'image_size': (IMG_MAX_SIZE, IMG_MAX_SIZE),
	'image_dimension': (IMG_MAX_SIZE, IMG_MAX_SIZE, CHANNELS),

	'multiplier': 24,

	'colors': {
		'rgb': [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
		'binary': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
	},

	'batch_size': BATCH_SIZE,

	'classification': {
		'loss': weighted_categorical_crossentropy(np.array([1, 2.5, 0.4, 1])),
		'metrics': ['accuracy', jacard_coef, dice_coef]
	},
	'segmentation': {
		'loss': jacard_coef_loss,
		'metrics': ['accuracy', jacard_coef, dice_coef]
	},

	'fit': {
		'epochs': 30,
		'shuffle': True,
		'batch_size': BATCH_SIZE,
		'callbacks': [checkpointer], # use checkpointer if you want to save the model
	}
}
