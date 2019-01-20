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

def clean_subfolders(folder):
    for folder in [make_path(folder, sub_folder) for sub_folder in os.listdir(folder)]:
        clean_folder(folder)

def remove_except_files(files):
    for file in files:
        if (file in config['except_files']):
            files.remove(file)
