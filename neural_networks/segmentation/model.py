#model file for segmentation
from keras.models import Model
from keras.layers.core import Activation
from keras.layers.merge import concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout

from config import config


def u_net():
    rows, cols, ch = config['image_dimension']
    inputs = Input((rows, cols, ch))
    neurons = config['neurons']
    
    c1 = Conv2D(neurons, (3, 3), activation = 'relu', padding = 'same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(neurons, (3, 3), activation = 'relu', padding = 'same') (c1)       
    p2 = MaxPooling2D(pool_size = (2, 2))(c1)

    c2 = Conv2D(2*neurons, (3, 3), activation = 'relu', padding = 'same') (p2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(2*neurons, (3, 3), activation = 'relu', padding = 'same') (c2)       
    p3 = MaxPooling2D(pool_size = (2, 2))(c2)

    c3 = Conv2D(4*neurons, (3, 3), activation = 'relu', padding = 'same') (p3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(4*neurons, (3, 3), activation = 'relu', padding = 'same') (c3)       
    p4 = MaxPooling2D(pool_size = (2, 2))(c3)
    
    c4 = Conv2D(8*neurons, (3, 3), activation = 'relu', padding = 'same') (p4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(8*neurons, (3, 3), activation = 'relu', padding = 'same') (c4)       
    p5 = MaxPooling2D(pool_size = (2, 2))(c4)
    
    c5 = Conv2D(16*neurons, (3, 3), activation = 'relu', padding = 'same') (p5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(16*neurons, (3, 3), activation = 'relu', padding = 'same') (c5)       
    p6 = MaxPooling2D(pool_size = (2, 2))(c5)    


    c6 = Conv2D(32*neurons, (3, 3), activation = 'relu', padding = 'same') (p6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32*neurons, (3, 3), activation = 'relu', padding = 'same') (c6) 

    up1 = concatenate([UpSampling2D(size = (2, 2))(c6), c5])
    up_c1 = Conv2D(16*neurons, (3, 3), activation = 'relu', padding = 'same') (up1)
    up_c1 = BatchNormalization()(up_c1)
    up_c1 = Conv2D(16*neurons, (3, 3), activation = 'relu', padding = 'same') (up_c1) 

    up2 = concatenate([UpSampling2D(size = (2, 2))(up_c1), c4])
    up_c1 = Conv2D(8*neurons, (3, 3), activation = 'relu', padding = 'same') (up2)
    up_c1 = BatchNormalization()(up_c1)
    up_c1 = Conv2D(8*neurons, (3, 3), activation = 'relu', padding = 'same') (up_c1)

    up3 = concatenate([UpSampling2D(size = (2, 2))(up_c1), c3])
    up_c2 = Conv2D(4*neurons, (3, 3), activation = 'relu', padding = 'same') (up3)
    up_c2 = BatchNormalization()(up_c2)
    up_c2 = Conv2D(4*neurons, (3, 3), activation = 'relu', padding = 'same') (up_c2)

    up4 = concatenate([UpSampling2D(size = (2, 2))(up_c2), c2])
    up_c3 = Conv2D(2*neurons, (3, 3), activation = 'relu', padding = 'same') (up4)
    up_c3 = BatchNormalization()(up_c3)
    up_c3 = Conv2D(2*neurons, (3, 3), activation = 'relu', padding = 'same') (up_c3)

    up5 = concatenate([UpSampling2D(size = (2, 2))(up_c3), c1])
    up_c4 = Conv2D(neurons, (3, 3), activation = 'relu', padding = 'same') (up5)
    up_c4 = BatchNormalization()(up_c4)
    up_c4 = Conv2D(neurons, (3, 3), activation = 'relu', padding = 'same') (up_c4)
    up_c4 = Dropout(0.2)(up_c4)
    
    conv_final = Conv2D(ch, (1, 1))(up_c4)
    conv_final = Activation('sigmoid')(conv_final)

    return Model(inputs, conv_final)
