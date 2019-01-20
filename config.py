from keras.callbacks import EarlyStopping, ModelCheckpoint
from platform import system as getSystem
import keras.backend as K
import numpy as np

IMG_MAX_SIZE = 224
BATCH_SIZE = 64



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

def dice_coef(y_true, y_pred, smooth = 100.0): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_inter = K.sum(y_true_f * y_pred_f)
    y_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2.0 * y_inter + smooth) / (y_sum + smooth)

def dice_coef_loss(y_true, y_pred, smooth = 100.0):
    return (1 - dice_coef(y_true, y_pred)) * smooth

earlystopper = EarlyStopping(monitor = 'val_jacard_coef', #stop when validation loss decreases
								min_delta = 0, #if val_loss < 0 it stops
								patience = 20, #minimum amount of epochs
								verbose = 1,
								mode = 'max') # print a text

checkpointer = ModelCheckpoint('model-polyp.h5', verbose = 1, save_best_only = True, monitor='val_jacard_coef', mode='max')


config = {
	'path_sep': '\\' if getSystem() == 'Windows' else '/',
	'except_files': ['.gitkeep'],

	'dtype': 'float32',

	'validation_split': 0.2,
	'test_split': 0.2,

	'image_max': IMG_MAX_SIZE,
	'image_size': (IMG_MAX_SIZE, IMG_MAX_SIZE),
	'image_dimension': (IMG_MAX_SIZE, IMG_MAX_SIZE, 3),

	'multiplier': 1,

	'color': {
		'rgb': {
			'red': (255, 0, 0),
			'green': (0, 255, 0),
			'blue': (0, 0, 255),
			'black': (0, 0, 0)
		},
		'binary': {
			'red': [1, 0, 0],
			'green': [0, 1, 0],
			'blue': [0, 0, 1],
			'black': [0, 0, 0]
		}
	},

	'batch_size': BATCH_SIZE,
	'loss': weighted_categorical_crossentropy(np.array([1,3,0.5,1])), #weighted_categorical_crossentropy(np.array([1,4,1,1])),#'categorical_crossentropy', #'dice_coef_loss', #'jacard_coef_loss',
	'metrics': ['accuracy', dice_coef, jacard_coef],
	'fit': {
		'steps_per_epoch': None, #1048//batch_size,
		'validation_steps': None, #128//batch_size,
		'epochs': 30,
		'shuffle': True,
		'batch_size': 10, #BATCH_SIZE,
		'class_weight': None,  #{0:1, 1:100, 2:1, 3:1}, #None, #(1,1,1,1),
		'callbacks': [earlystopper, checkpointer], #, checkpointer], # use checkpointer if you want to save the model
	},
	'class_weight': (1,1,1,1)
}
