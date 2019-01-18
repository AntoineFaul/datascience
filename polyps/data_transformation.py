from platform import system as getSystem
from polyps.file_manager import make_path, load_image, clean_folder
import cv2 as cv
from PIL import Image
import numpy as np
import glob
import os



def pixelsVerification(picture):
    (rows, cols, channels) = np.shape(picture)
    toReturn = np.empty(shape=(rows, cols, channels))

    for i in range( channels):
        plop, toReturn[:, :, i] = cv.threshold(picture[:, :, i], 0.5, 1, cv.THRESH_BINARY)

    return toReturn
        
def createMask(folderIn, folderOut, masks, masks_namesPrefixe = None):
    # Suppresion of the files inside the output folder
    clean_folder(folderOut)
    names = os.listdir(folderIn)
    names.remove('.gitkeep')

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

        print("Picture " + str(cpt))
        cpt += 1

    return(store_im)
 
def create_binary_masks():
    # folders
    fileIn = make_path('polyps', 'input', 'label')
    fileOut = make_path('polyps', 'output', 'label')
    
    # mask to create 
    masks = np.array([[0, 0, 0], #background = black
                        [1, 0, 0], #Polype = red
                        [0, 1, 0], #Wall = green
                        [0, 0, 1]]) #Dirt = blue

    # create the pictures
    # if the masks_namesPrefixe is not define, the prefix of the pictures will
    # be the index of the mask in the Array masks
    # /!\ All of files of the Output folder will be delete before.
    plop = createMask(folderIn = fileIn, folderOut = fileOut, masks = masks)
    chunks = [np.swapaxes(np.array(plop[x:x+4]), 0, 2) for x in range(0, len(plop), 4)]  
    
    print("END\n")
    return(chunks)
