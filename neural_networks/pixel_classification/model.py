from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dropout, Lambda
from config import config
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Lambda
from keras.layers.core import SpatialDropout2D, Activation



def u_net(num_classes = 3, IMG_SIZE = config['image_dimension']):
    # Build U-Net model
    inputs = Input(IMG_SIZE)
    s = BatchNormalization()(inputs) #Ioffe and Szegedy, 2015
    s = Dropout(0.5)(s)
    neurons = 64 #original 64
    
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
    
    outputs = Conv2D(4, (1, 1), activation='softmax') (c9)  #each output row will sum up to 1
    model = Model(inputs=[inputs], outputs=[outputs])
#    model.summary()
    return(model)
    
def u_net_batch_norm_upc(output_channels =4, act = 'softmax'):
    inputs = Input((224, 224, 3))
    neurons = 64
    #s = BatchNormalization()(inputs) #Ioffe and Szegedy, 2015
    #s = Dropout(0.15)(s)
    
    c1 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (c1)       
    p2 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (p2)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (c2)       
    p3 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (p3)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (c3)       
    p4 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (p4)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (c4)       
    p5 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (p5)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (c5)       
    p6 = MaxPooling2D(pool_size=(2, 2))(c5)    


    c6 = Conv2D(32*neurons, (3, 3), activation='relu', padding='same') (p6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(32*neurons, (3, 3), activation='relu', padding='same') (c6) 

    up1 = concatenate([UpSampling2D(size=(2, 2))(c6), c5])
    up_c1 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (up1)
    up_c1 = BatchNormalization()(up_c1)
    up_c1 = Conv2D(16*neurons, (3, 3), activation='relu', padding='same') (up_c1) 

    up2 = concatenate([UpSampling2D(size=(2, 2))(up_c1), c4])
    up_c1 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (up2)
    up_c1 = BatchNormalization()(up_c1)
    up_c1 = Conv2D(8*neurons, (3, 3), activation='relu', padding='same') (up_c1)

    up3 = concatenate([UpSampling2D(size=(2, 2))(up_c1), c3])
    up_c2 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (up3)
    up_c2 = BatchNormalization()(up_c2)
    up_c2 = Conv2D(4*neurons, (3, 3), activation='relu', padding='same') (up_c2)

    up4 = concatenate([UpSampling2D(size=(2, 2))(up_c2), c2])
    up_c3 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up4)
    up_c3 = BatchNormalization()(up_c3)
    up_c3 = Conv2D(2*neurons, (3, 3), activation='relu', padding='same') (up_c3)

    up5 = concatenate([UpSampling2D(size=(2, 2))(up_c3), c1])
    up_c4 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (up5)
    up_c4 = BatchNormalization()(up_c4)
    up_c4 = Conv2D(neurons, (3, 3), activation='relu', padding='same') (up_c4)
    up_c4 = Dropout(0.4)(up_c4)
    
    conv_final = Conv2D(output_channels, (1, 1))(up_c4)
    conv_final = Activation(act)(conv_final)

    model = Model(inputs, conv_final)
#    model.summary()
    return model


