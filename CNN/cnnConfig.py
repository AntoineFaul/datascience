from platform import system as getSystem
import polyps.file_manager as fm

# CV    -> learnOpenCV_model
# RM    -> reduced_model
# FM    -> final_model
# RDM   -> reduced_dropOut_model
# RGM   -> reduced_generator_model


def getConfig(name):
    if name == "CV":
        return learnOpenCV_model
    if name == "RM":
        return reduced_model
    if name == "FM":
        return final_model
    if name == "RDM":
        return reduced_dropOut_model
    if name == "RGM":
        return reduced_generator_model
    return default_model


default_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "default_model",

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
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
        'shuffle': False,
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

    'callbacks1': {
        'monitor': 'val_acc',
        'min_delta': -0.25,
        'patience': 1,
        'verbose': 0,
        'mode': 'auto',  # 'max'
        'baseline': None,
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
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
        'shuffle': False,  # True or False
    },
}

reduced_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "reduced_model",

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
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
        'shuffle': False,
    },
}

reduced_generator_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "reduced_generator_model",

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
        'epochs': 100,
        'verbose': 1,
        'generator': 'generator',  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
        'shuffle': False,
    },
}

reduced_dropOut_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "reduced_dropOut_model",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': [
        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (5, 5), 'padding': 'same', 'activation': 'relu', 'shape': 'shape'},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.01},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.01},

        {'index': 'Conv2D', 'filters': 5,
            'kernel_size': (3, 3), 'padding': 'same', 'activation': 'relu', 'shape': None},
        {'index': 'MaxPooling2D', 'pool_size': (2, 2)},
        {'index': 'Dropout', 'rate': 0.01},

        {'index': 'Flatten'},
        {'index': 'Dense', 'units': 5, 'activation': 'relu'},
        {'index': 'Dropout', 'rate': 0.05},
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
        'epochs': 100,
        'verbose': 1,
        'generator': None,  # 'generator',  # None
        'callbacks': None,  # 'callbacks', # None
        'shuffle': False,
    },
}

final_model = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "final_model",

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
