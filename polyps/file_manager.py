from PIL import Image
import numpy as np
import os
from config import config


def make_path(*args):
	return config['path_sep'].join(args)

def load_image(filename):
    im = Image.open(filename)
    # conversion to numpy array
    # each pixels are represented by floating points
    im = np.array(im, dtype = config['dtype'])
    im /= 255

    return im

def clean_folder(forlder):
	# Suppresion of the files inside the output folder
    fileList = os.listdir(forlder)

    for filename in fileList:
        if (not(filename in config['except_files'])):
            os.remove(forlder + config['path_sep'] + filename)

def clean_folder_group(folder_group, *sub_folders):
    for folder in [make_path(folder_group, sub_folder) for sub_folder in sub_folders]:
        clean_folder(folder)
