from platform import system as getSystem
import polyps.file_manager as fm

# DM    -> defaultModel
# DMG   -> defaultModelWithGenerator


def getConfig(name):
    if name == "DM":
        return defaultModel
    if name == "DMG":
        return defaultModelWithGenerator
    else:
        return defaultModel


defaultModel = {
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

defaultModelWithGenerator = {
    'path_sep': '\\' if getSystem() == 'Windows' else '/',

    # Input
    'folderTest': fm.make_path("polyps", "cnn_dataAugmented", "TestAugmented"),
    'folderTrain': fm.make_path("polyps", "cnn_dataAugmented", "Train"),

    # Ouptut
    'folderModel': fm.make_path("CNN", "modelSave"),
    'nameModel': "defaultModelWithGenerator",

    # DataSet loading
    'maxsize': (64, 64),
    'perCent_Test': 0.2,
    'perCent_Validation': 0.2,

    'model': {
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
