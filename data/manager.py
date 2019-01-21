#this file consists the organization functions like creating paths etc.
from PIL import Image 
import numpy as np
import os

from config import config


def make_path(*args):#create an absolute path out of the relative paths
	return config['path_sep'].join(args)

def load_image(filename):# load the image and normalize
    img = Image.open(filename)
    # conversion to numpy array
    # each pixels are represented by floating points
    img = np.array(img, dtype = config['dtype'])
    img /= 255

    return img

def load_images(folder):
    data = []

    for filename in list_dir(folder):
        data.append(load_image(make_path(folder, filename)))

    return data

def clean_folder(forlder):
	# Suppresion of the files inside the output folder
    fileList = os.listdir(forlder)

    for filename in fileList:
        if (not(filename in config['except_files'])):
            os.remove(forlder + config['path_sep'] + filename)

def clean_subfolders(folder):
    for folder in [make_path(folder, sub_folder) for sub_folder in os.listdir(folder)]:
        clean_folder(folder)

def list_dir(folder):
    file_list = os.listdir(folder)
    remove_except_files(file_list)

    return file_list

def remove_except_files(files):
    for file in files:
        if (file in config['except_files']):
            files.remove(file)

def load_imgs():
    img_train = np.array(load_images(make_path('polyps', 'training', 'data')), dtype = config['dtype'])
    img_val = np.array(load_images(make_path('polyps', 'validation', 'data')), dtype = config['dtype'])
    img_test = np.array(load_images(make_path('polyps', 'test', 'data')), dtype = config['dtype'])

    return (img_train, img_val, img_test)
