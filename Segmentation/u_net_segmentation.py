from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
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
    
    c2 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (inputs)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (c2)       
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (c3)       
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (c4)       
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (c5)       
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    
    c6 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (p5)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (c6)       
    p6 = MaxPooling2D(pool_size=(2, 2))(c6)    


    c7 = Conv2D(32*neurons, (3, 3), activation='relu', padding='same') (p6)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(32*neurons, (3, 3), activation='relu', padding='same') (c7) 

    up1 = concatenate([UpSampling2D(size=(2, 2))(c7), c6])
    up_c1 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (up1)
    up_c1 = BatchNormalization()(up_c1)
    up_c1 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (up_c1) 

    up2 = concatenate([UpSampling2D(size=(2, 2))(up_c1), c5])
    up_c2 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (up2)
    up_c2 = BatchNormalization()(up_c2)
    up_c2 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (up_c2)

    up3 = concatenate([UpSampling2D(size=(2, 2))(up_c2), c4])
    up_c3 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (up3)
    up_c3 = BatchNormalization()(up_c3)
    up_c3 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (up_c3)

    up4 = concatenate([UpSampling2D(size=(2, 2))(up_c3), c3])
    up_c4 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up4)
    up_c4 = BatchNormalization()(up_c4)
    up_c4 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up_c4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(up_c4), c2])
    up_c5 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up5)
    up_c5 = BatchNormalization()(up_c5)
    up_c5 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up_c5)
    up_c5 = Dropout(0.2)(up_c5)
    
    conv_final = Conv2D(chanels, (1, 1))(up_c5)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")
#    model.summary()
    return model


