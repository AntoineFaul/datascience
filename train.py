import model
from polyps import data_transformation
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import sys


def load_transform_pictures(folder):
    maxsize = (224,224)
    x_train=[]
    for filename in glob.glob(folder):
        im=Image.open(filename)
        im.thumbnail(maxsize) # find other technique
        im = np.array(im,dtype="float32")
        im = im /255 # for learing faster
        x_train.append(im)
    return(x_train)

    
if __name__ == "__main__":
    batch_size = 64
    model = model.u_net() #what does the Adam optimizer do
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy' , metrics = ['accuracy'])#,pixel_accuracy])
    if sys.platform == 'linux':
        im = np.array(load_transform_pictures('polyps/data/*.jpg'))
        test = np.array(load_transform_pictures('polyps/test/*.jpg'))
	        
        print(im)
        print("--------------------------------------")
        print(test)    
    else:
        im = np.array(load_transform_pictures('C:\\Users\\MaxSchemmer\\Documents\\Data\\polyps_test_2\\Data\\*.jpg'))
        test = np.array(load_transform_pictures('C:\\Users\\MaxSchemmer\\Documents\\Data\\polyps_test_2\\Data\\test\\*.jpg'))
    mask = np.array(data_transformation.create_binary_masks()) 
    print(mask)
    model.fit(x = im,y=mask,
                        steps_per_epoch = 1048//batch_size,#1048//batch_size,
                        validation_split = 0.2,
                        validation_steps = 128//batch_size,
                        epochs = 1, 
                        )
    

    lab_pred = model.predict(test)

