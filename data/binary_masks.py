#this file creates from the colorized images binary versions per class
import cv2
import numpy as np

from .manager import make_path, load_image, list_dir
from config import config


def pixelsVerification(picture): #check the value of the pixels 
    (rows, cols, channels) = np.shape(picture)
    toReturn = np.empty(shape = (rows, cols, channels))

    for i in range(channels):
        plop, toReturn[:, :, i] = cv2.threshold(picture[:, :, i], 0.5, 1, cv2.THRESH_BINARY)

    return toReturn
        
def createMask(folderIn, masks):#create the binary masks
    store_im = []
    cpt = 0

    for filename in [make_path(folderIn, name) for name in list_dir(folderIn)]:
        picture = pixelsVerification(load_image(filename))

        for i in range(len(masks)):
            store_im.append(cv2.inRange(picture, masks[i], masks[i]) / 255)

        print("\rPicture - " + str(cpt), end = '')
        cpt += 1

    return store_im
 
def create(path):
    print("\rCreate Binary Masks from folder: " + path)

    masks = np.array(config['colors']['binary'])

    # create the pictures
    plop = createMask(path, masks)
    chunks = [np.swapaxes(np.array(plop[x : x+4]), 0, 2) for x in range(0, len(plop), 4)]  

    print("\rDone. (Nb = " + str(len(chunks)) + ")\n")
    return chunks
