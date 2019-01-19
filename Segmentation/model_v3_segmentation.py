import u_net_segmentation as us
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt


def dice_coef(y_true, y_pred): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jacard_coef(y_true, y_pred): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_transform_pictures(folder):
    maxsize = (224,224)
    x_train=[]
    for filename in glob.glob(folder):
        im=Image.open(filename)
        im.thumbnail(maxsize)
        im = np.array(im)
        im = im / 255        
        x_train.append(im)
    return(x_train)
    
if __name__ == '__main__':
    batch_size = 128
    model = us.u_net_segmentation(chanels = 3)
    model.compile(optimizer = Adam(lr = 1e-4), loss = jacard_coef, metrics = ['accuracy', dice_coef, jacard_coef])
    im = np.array(load_transform_pictures('polyps\\input\\data\\*.jpg'))
    mask = np.array(load_transform_pictures('polyps\\input\\label\\*.jpg'))
    model.fit(x = im,
                y=mask,
                batch_size = batch_size,
                epochs = 20, 
                validation_split= 0.2
            )
    
    lab_pred = model.predict(im[:1])
    plt.imshow(lab_pred[0])
    plt.show()
