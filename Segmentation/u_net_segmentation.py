from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Lambda
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.utils.data_utils import get_file
def u_net_segmentation(chanels =3):
    inputs = Input((224, 224, chanels))
    neurons = 64
    
    c1 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (inputs)
    c1 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (c1)       
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (c2)       
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (c3)       
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (c4)       
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (c5)       
  
    
    up1 = Conv2DTranspose(neurons*8, (2, 2), strides=(2, 2), padding='same') (c5)#
    up1 = concatenate([up1,c4])
    up_c1 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (up1)
    up_c1 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (up_c1)

    up2 = Conv2DTranspose(neurons*4, (2, 2), strides=(2, 2), padding='same') (up_c1)#
    up2 = concatenate([up2,c3])
    up_c2 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (up2)
    up_c2 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (up_c2)

    up3 = Conv2DTranspose(neurons*2, (2, 2), strides=(2, 2), padding='same') (up_c2)#
    up3 = concatenate([up3,c2])
    up_c3 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up3)
    up_c3 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up_c3)

    up4 = Conv2DTranspose(neurons, (2, 2), strides=(2, 2), padding='same') (up_c3)#
    up4 = concatenate([up4,c1])
    up_c4 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (up4)
    up_c4 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (up_c4)
    up_c4 = Dropout(0.2)(up_c4)
    
    conv_final = Conv2D(chanels, (1, 1))(up_c4)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final)
#    model.summary()
    return model


