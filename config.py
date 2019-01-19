from keras.callbacks import EarlyStopping, ModelCheckpoint
from platform import system as getSystem
import keras.backend as K

IMG_MAX_SIZE = 256
BATCH_SIZE = 32


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

earlystopper = EarlyStopping(monitor = 'val_loss', #stop when validation loss decreases
								min_delta = 0, #if val_loss < 0 it stops
								patience = 10, #minimum amount of epochs
								verbose = 1) # print a text

checkpointer = ModelCheckpoint('model-polyp.h5', verbose = 1, save_best_only = True)


config = {
	'path_sep': '\\' if getSystem() == 'Windows' else '/',
	'except_files': ['.gitkeep'],

	'dtype': 'float32',

	'validation_split': 0.2,
	'test_split': 0.2,

	'image_max': IMG_MAX_SIZE,
	'image_size': (IMG_MAX_SIZE, IMG_MAX_SIZE),
	'image_dimension': (IMG_MAX_SIZE, IMG_MAX_SIZE, 3),

	'multiplier': 4,
	'seed': 1,
	'augmentation': {
		'zoom_range': 0.0,
		'height_shift_range': 0.0,
		'width_shift_range': 0.0,
		'rotation_range': 0, #90,
		'brightness_range': (0.75,1.25),
		'flip': {
			'horizontal': True,
			'vertical': True
		}
	},

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
	'loss': 'categorical_crossentropy', #'dice_coef_loss', #'jacard_coef_loss',
	'metrics': ['accuracy', dice_coef, jacard_coef],
	'fit': {
		'steps_per_epoch': None, #1048//batch_size,
		'validation_steps': None, #128//batch_size,
		'epochs': 1,
		'shuffle': True,
		'batch_size': None, #BATCH_SIZE,
		'class_weight': None, #(1, 1 , 1, 1),
		'callbacks': [earlystopper], #, checkpointer], # use checkpointer if you want to save the model
	}
}
