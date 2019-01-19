from platform import system as getSystem
from polyps.file_manager import make_path, load_image, clean_folder, remove_except_files
import cv2 as cv
from PIL import Image
import numpy as np
import glob
import os
from config import config


def pixelsVerification(picture):
    (rows, cols, channels) = np.shape(picture)
    toReturn = np.empty(shape = (rows, cols, channels))

    for i in range(channels):
        plop, toReturn[:, :, i] = cv.threshold(picture[:, :, i], 0.5, 1, cv.THRESH_BINARY)

    return toReturn
        
def createMask(folderIn, folderOut, masks, masks_namesPrefixe = None):
    # Suppresion of the files inside the output folder
    clean_folder(folderOut)
    names = os.listdir(folderIn)
    remove_except_files(names)

    # Configuration of the output files prefixes
    if masks_namesPrefixe and len(masks) == len(masks_namesPrefixe):
        prefixes = masks_namesPrefixe
    else:
        prefixes = [str(i) for i in range(len(masks))]

    store_im = []
    cpt = 0

    for filename in glob.glob(make_path(folderIn, '*.jpg')):
        picture = pixelsVerification(load_image(filename))
        
        for i in range(len(masks)):
            imOut = cv.inRange(picture, masks[i], masks[i])
            cv.imwrite(make_path(folderOut, prefixes[i] + '_' + names[cpt]), imOut)
            store_im.append(imOut/255)

        print("\rPicture - " + str(cpt), end = '')
        cpt += 1

    return(store_im)
 
def create_binary_masks(path):
    fileOut = make_path('polyps', 'output', 'binary')

    print("\rCreate Binary Masks from folder: " + path)
    
    # mask to create 
    masks = np.array([config['color']['binary']['black'],
                        config['color']['binary']['red'],
                        config['color']['binary']['green'],
                        config['color']['binary']['blue']])

    # create the pictures
    # if the masks_namesPrefixe is not define, the prefix of the pictures will
    # be the index of the mask in the Array masks
    # /!\ All of files of the Output folder will be delete before.
    plop = createMask(folderIn = path, folderOut = fileOut, masks = masks)
    chunks = [np.swapaxes(np.array(plop[x:x+4]), 0, 2) for x in range(0, len(plop), 4)]  

    print("\rDone. (Nb = " + str(len(chunks)) + ")\n")
    return(chunks)
