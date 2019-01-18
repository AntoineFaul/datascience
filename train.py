from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from polyps import data_augmentation, data_transformation, file_manager as fm
from keras.optimizers import Adam
from PIL import Image
from keras import backend as K #use tensors
import numpy as np
import glob
import numpy as np
from PIL import Image
from keras.optimizers import Adam
import model
import matplotlib.pyplot as plt
import keras.backend as K


def load_transform_pictures(folder):
    x_train = []

    for filename in glob.glob(folder):
        im = fm.load_image(filename)
        x_train.append(im)

    return(x_train)

def pixel_class(c):
    if c == 1:
        return(255, 0, 0)
    elif c == 2:
        return(0, 255, 0)
    elif c == 3:
        return(0, 0, 255)
    else:
        return(0, 0, 0)
        
def find_class(c):
    return c.argmax()

def merge(array):
    final_images = []

    for image in array:
        cimage = []

        for row in image:
            crow = []

            for pixel in row:
                crow.append(pixel_class(find_class(pixel)))

            cimage.append(crow)

        final_images.append(cimage)

    return(final_images)

def write_image(array, directory):
    index = 0
    img_store =[]
    for image in array:
        index = index +1
        img = Image.new("RGB", (224,224), "white")

        for i in range(224):
            for j in range(224):
                img.putpixel((i,j),image[j][i])

        name = '{0:04}'.format(index) + "_output.jpg"
        img.save(fm.make_path(directory,name))
        img_store.append(np.array(img))
    return img_store

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

if __name__ == "__main__":
    data_augmentation.execute()

    batch_size = 64
    model = model.u_net(IMG_SIZE = (224,224,3)) #what does the Adam optimizer do

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy' , metrics = ['accuracy',dice_coef,jacard_coef])# old learning rate 1e-4, pixel_accuracy])

    im = np.array(load_transform_pictures(fm.make_path('polyps', 'input', 'data', '*.jpg')))
    test = np.array(load_transform_pictures(fm.make_path('polyps', 'test','data', '*.jpg')))
    output = fm.make_path('polyps', 'output')
    path = fm.make_path('polyps', 'input', 'label')
    path_test = fm.make_path('polyps', 'test', 'label')
    mask = np.array(data_transformation.create_binary_masks(path=path)) 
    mask_test = np.array(data_transformation.create_binary_masks(path = path_test))
#    checkpointer = ModelCheckpoint('model-polyp.h5', verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', #stop when validation loss decreases
                                 min_delta=0, #if val_loss < 0 it stops
                                 patience=10, #minimum amount of epochs
                                 verbose=1) # print a text
    model.fit(x = im,y=mask,
                        validation_split = 0.2,
#                        steps_per_epoch = 1048//batch_size,
                        epochs = 1,
                        batch_size=batch_size,
#                        validation_steps = 128//batch_size,
                        callbacks =[earlystopper
#                                    ,checkpointer #if you want to save the model
                                   ]     
                        )
    
    lab_pred = model.predict(test, verbose=1)
    evaluate = model.evaluate(x=test, y=mask_test,batch_size=batch_size)
    display_im = write_image(merge(lab_pred),output)
    plt.imshow(display_im[0])#plots the first picture
    print("Evaluation : Loss: "+ str(evaluate[0])+", Accuracy: " + str(evaluate[1])+", Dice Coefficient: " + str(evaluate[2])+", Jacard Coefficient: " + str(evaluate[3]))
    

