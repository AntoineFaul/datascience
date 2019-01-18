import cv2 as cv
import numpy as np
import glob
import os
import sys

from PIL import Image

if sys.platform == 'linux':
    path_separator = '/'
else:
    path_separator = '\\'

def pixelsVerification( picture):
    
    (rows, cols, channels) = np.shape(picture)
    toReturn = np.empty(shape=( rows, cols, channels))
    for i in range( channels):
        plop, toReturn[:,:,i] = cv.threshold( picture[:,:,i], 0.5, 1, cv.THRESH_BINARY)

    return toReturn
        
def createMask( folderIn, folderOut, masks, masks_namesPrefixe = None):
    maxsize = (224,224)
    # Suppresion of the files inside the output folder
    fileList = os.listdir(folderOut)
    for fileName in fileList:
        os.remove(folderOut+path_separator+fileName)
    
    names = os.listdir(folderIn)
    # Configuration of the output files prefixes
    if masks_namesPrefixe and len(masks)==len(masks_namesPrefixe):
        prefixes = masks_namesPrefixe
    else:
        prefixes = [str(i) for i in range(len(masks))]
    store_im = []
    cpt = 0
    for filename in glob.glob(folderIn+path_separator+"*.jpg"):
        im = Image.open(filename)
        im.thumbnail(maxsize)
        # conversion to numpy array 
        im = np.array(im)
        # each pixels are represented by floating points
        im = im.astype('float32')
        im /= 255
        
        picture = pixelsVerification( im)
        
        for i in range(len(masks)):
            imOut = cv.inRange(	picture, masks[i], masks[i])
            #print( "out : " + folderOut + "\\" + prefixes[i] + "_" + names[cpt] +"\n")
            cv.imwrite( folderOut + path_separator + prefixes[i] + "_" + names[cpt], imOut)
            imOut = imOut/255
            store_im.append(imOut)
        print("Picture " + str(cpt))
        cpt += 1
    return(store_im)
 
def create_binary_masks():
    
    # folders
    if sys.platform == 'linux':
        fileIn = 'polyps/input/label/'
        fileOut = 'polyps/output/label'
    else:
        fileIn = "polyps\\input\\label"
        fileOut = "polyps\\output\\label"
    
    # mask to create 
    masks = np.array([[0,0,0], #background = black
                      [1,0,0], #Polype = red
                      [0,1,0], #Wall = green
                      [0,0,1]]) #Dirt = blue
    

    # create the pictures
    # if the masks_namesPrefixe is not define, the prefix of the pictures will
    # be the index of the mask in the Array masks
    # /!\ All of files of the Output folder will be delete before.
    plop =  createMask( folderIn=fileIn, folderOut=fileOut, masks = masks)
    print(plop)
    chunks = [np.swapaxes(np.array(plop[x:x+4]),0,2) for x in range(0, len(plop), 4)]  
    
    
    print("END\n")
    return(chunks)
