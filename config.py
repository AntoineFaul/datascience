import keras.backend as K

IMG_MAX_SIZE = 256
BATCH_SIZE = 64


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


config = {
	'path_sep': '\\' if getSystem() == 'Windows' else '/',
	'except_file': '.gitkeep',

	'dtype': 'float32',

	'validation_split': 0.2,
	'test_split': 0.2,

	'image_max': IMG_MAX_SIZE,
	'image_size': (IMG_MAX_SIZE, IMG_MAX_SIZE),
	'image_dimension': (IMG_MAX_SIZE, IMG_MAX_SIZE, 3),

	'multiplier': 1
	'class_data': 'data',
	'class_label': 'label',
	'seed': 1,
	'augmentation': {
		'brightness_range': (0.75,1.25),
		'flip': {
			'horizontal': True,
			'vertical': True
		},
		'rotation_range': 0, #90,
		'height_shift_range': 0.0,
		'width_shift_range': 0.0,
		'zoom_range': 0.0
	},

	'color': {
		'red': (255, 0, 0),
		'green': (0, 255, 0),
		'blue': (0, 0, 255),
		'black': (0, 0, 0)
	},

	'batchsize': BATCH_SIZE,
	'loss': 'categorical_crossentropy', #'dice_coef_loss', #'jacard_coef_loss',
	'metrics': ['accuracy', dice_coef, jacard_coef],
	'validation_step': None, #128//batch_size
	'epochs': 1
}
