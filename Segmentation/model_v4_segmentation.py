
import glob
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import *
from keras.layers.merge import concatenate
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from PIL import Image

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def u_net(num_classes=3,IMG_SIZE = (224,224)):
    # Build U-Net model
    inputs = Input(IMG_SIZE+(3,)) # +3 = RGB image 
    s = BatchNormalization()(inputs) #Ioffe and Szegedy, 2015
    s = Dropout(0.5)(s)
    neurons = 32 #original 64
    
    c1 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (s)#(inputs)#
    c1 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (c1)#original no padding
    p1 = MaxPooling2D((2, 2)) (c1)#original with stride 2
    
    c2 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same') (p1)
#    c2 = BatchNormalization()(c2)
    c2 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(neurons*16, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(neurons*16, (3, 3), activation='relu', padding='same') (c5)
    
    u6 = Conv2DTranspose(neurons*8, (2, 2), strides=(2, 2), padding='same') (c5)#why strides = 2
    u6 = concatenate([u6, c4])
    c6 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same') (c6)
    
    u7 = Conv2DTranspose(neurons*4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same') (c7)
    
    u8 = Conv2DTranspose(neurons*2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same') (c8)
    
    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (c9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid') (c9) 
    model = Model(inputs=[inputs], outputs=[outputs])
#    model.summary()
    return(model)

def load_transform_pictures(folder):
    maxsize = (224,224)
    x_train=[]
    for filename in glob.glob(folder):
        im=Image.open(filename)
        im.thumbnail(maxsize)
        im = np.array(im)
        im = im /255
        x_train.append(im)
    return(x_train)
 
if __name__ == '__main__':
    batch_size = 64
    model = u_net() #what does the Adam optimizer do
    model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss , metrics = ['accuracy'])
    im = np.array(load_transform_pictures('C:\\Users\\MaxSchemmer\\Documents\\git\\datascience_v3\\polyps\\input\\data\\*.jpg'))
    mask = np.array(load_transform_pictures('C:\\Users\\MaxSchemmer\\Documents\\git\\datascience_v3\\polyps\\input\\label\\*.jpg'))
    model.fit(x = im,y=mask,
                        steps_per_epoch = 1048//batch_size,#1048//batch_size,
                        validation_split = 0.2,
                        validation_steps = 128//batch_size,
                        epochs = 1, 
                        )
    
    lab_pred = model.predict(im)
    plt.imshow(lab_pred[0])

