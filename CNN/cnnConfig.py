from platform import system as getSystem
import polyps.file_manager as fm

# DM    -> default_model
# DMG   -> defaultGenerator_model
# CV    -> learnOpenCV_model
# V2    -> defaultV2_model
# V3    -> defaultV3_model
# V3C   -> defaultV3_callbacks_model
# V3S   -> defaultV3_shuffle_model
# V4    -> defaultV4_model
# V5    -> defaultV5_model


def getConfig(name):
    if name == "DM":
        return default_model
    if name == "DMG":
        return defaultGenerator_model
    if name == "CV":
        return learnOpenCV_model
    if name == "V2":
        return defaultV2_model
    if name == "V3":
        return defaultV3_model
    if name == "V3C":
        return defaultV3_callbacks_model
    if name == "V3S":
        return defaultV3_shuffle_model
    if name == "V4":
        return defaultV4_model
    if name == "V5":
        return defaultV5_model
    return default_model


default_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "defaulModel",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'softmax'},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 0,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
    },
}

defaultGenerator_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "defaultGenerator_model",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'softmax'},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 0,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
    },
}

learnOpenCV_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "learnOpenCV_model",

    # DataSet loading
    'maxsize': (32, 32),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 32,
         'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu',
         'shape': 'shape'},
        {'index': 'Conv2D', 'filters': 32,
         'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu',
         'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.25},

        {'index': 'Conv2D', 'filters': 64,
         'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu',
         'shape': None},
        {'index': 'Conv2D', 'filters': 64,
         'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu',
         'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.25},

        {'index': 'Conv2D', 'filters': 64,
         'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu',
         'shape': None},
        {'index': 'Conv2D', 'filters': 64,
         'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu',
         'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.25},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 512, 'activation': 'softmax'},
        {'index': 'Dropout', 'rate': 0.5},
        {'index': 'Dense', 'units': 'nClasses', 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 0,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
    },
}

defaultV2_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "v2Model",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'relu'},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 0,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
    },
}

defaultV3_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "modelV3",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'relu'},
        {'index': 'Dropout', 'rate': 0.25},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 0,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
    },
}

defaultV3_callbacks_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "modelV3_callbacks",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'relu'},
        {'index': 'Dropout', 'rate': 0.25},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks1': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 2,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    'callbacks2': {
        'monitor': 'val_acc',
        'min_delta': 0,
        'patience': 1,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': 2,  # None
        'suffle': False,  # None (n)
    },
}

defaultV3_shuffle_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "modelV3_shuffle",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.1},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'relu'},
        {'index': 'Dropout', 'rate': 0.25},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks1': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 2,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    'callbacks2': {
        'monitor': 'val_acc',
        'min_delta': 0,
        'patience': 1,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # None, 1 or 2
        'shuffle': False,  # True or False
    },
}

defaultV4_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "modelV4",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.05},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.05},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.05},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'relu'},
        {'index': 'Dropout', 'rate': 0.1},
        {'index': 'Dense', 'units': 2, 'activation': 'sigmoid'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks1': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 2,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    'callbacks2': {
        'monitor': 'val_acc',
        'min_delta': 0,
        'patience': 1,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # None, 1 or 2
        'shuffle': False,  # True or False
    },
}

defaultV5_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "modelV5",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 3,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},

        # {'index': 'Conv2D', 'filters': 3,
        #    'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        #{'index': 'MaxPooling2D', 'pool_size': (2, 2)},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 2, 'activation': 'softmax'}
    ],

    'compile': {
        'optimizer': 'rmsprop',
        'loss': 'categorical_crossentropy',
        'metrics': ['accuracy'],
    },

    'callbacks1': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 2,
        'mode': 'auto',  # 'max'
        'baseline': 0.8,
        'restore': False,
    },

    'callbacks2': {
        'monitor': 'val_acc',
        'min_delta': 0,
        'patience': 250,
        'verbose': 1,
        'mode': 'auto',  # 'max'
        'baseline': None,
        'restore': False,
    },

    # Generator used during the trainning
    'generator': {
        'rotation': 0,  # Degree range for random rotations
        'zoom': 0.0,  # (0.8, 1) # Range for random zoom (<1 : zoom in)
        'width_shift': 0.0,  # 0.01 # shift by fraction of total width
        'height_shift': 0.0,  # 0.01 # shift by fraction of total height
        'brightness': None,
        # (0.75, 1.25), # Range for random modification of the brightness
        'horizontal_flip': True,  # randomly flip images
        'vertical_flip': True,  # randomly flip images
    },

    'fit': {
        'batch_size': 224,
        'epochs': 40,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # None, 1 or 2
        'shuffle': False,  # True or False
    },
}
