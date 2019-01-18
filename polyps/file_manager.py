from platform import system as getSystem
from PIL import Image
import numpy as np
import os

PATH_SEP = '\\' if getSystem() == 'Windows' else '/'


def make_path(*args):
	return PATH_SEP.join(args)

def load_image(filename):
    im = Image.open(filename)
    # conversion to numpy array
    # each pixels are represented by floating points
    im = np.array(im, dtype="float32")
    im /= 255

    return im

def clean_folder(forlder):
	# Suppresion of the files inside the output folder
    fileList = os.listdir(forlder)

    for filename in fileList:
        if (filename != '.gitkeep'):
            os.remove(forlder+PATH_SEP+filename)
