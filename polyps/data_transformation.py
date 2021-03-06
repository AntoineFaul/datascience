import cv2
import glob
import numpy as np
from polyps.file_manager import make_path, load_image, clean_folder, list_dir
from config import config


def pixelsVerification(picture):
    (rows, cols, channels) = np.shape(picture)
    toReturn = np.empty(shape = (rows, cols, channels))

    for i in range(channels):
        plop, toReturn[:, :, i] = cv2.threshold(picture[:, :, i], 0.5, 1, cv2.THRESH_BINARY)

    return toReturn
        
def createMask(folderIn, masks):
    store_im = []
    cpt = 0

    for filename in [make_path(folderIn, name) for name in list_dir(folderIn)]:
        picture = pixelsVerification(load_image(filename))

        for i in range(len(masks)):
            store_im.append(cv2.inRange(picture, masks[i], masks[i]) / 255)

        print("\rPicture - " + str(cpt), end = '')
        cpt += 1

    return store_im
 
def create_binary_masks(path):
    print("\rCreate Binary Masks from folder: " + path)
    
    # mask to create 
    masks = np.array([config['color']['binary']['black'],
                        config['color']['binary']['red'],
                        config['color']['binary']['green'],
                        config['color']['binary']['blue']])

    # create the pictures
    plop = createMask(folderIn = path, masks = masks)
    chunks = [np.swapaxes(np.array(plop[x:x+4]), 0, 2) for x in range(0, len(plop), 4)]  

    print("\rDone. (Nb = " + str(len(chunks)) + ")\n")
    return chunks
