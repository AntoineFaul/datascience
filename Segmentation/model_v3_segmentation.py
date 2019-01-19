import u_net_segmentation as us
from keras import backend as K
from keras.optimizers import Adam
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt


def dice_coef(y_true, y_pred, smooth = 100.0): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_inter = K.sum(y_true_f * y_pred_f)
    y_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (2.0 * y_inter + smooth) / (y_sum + smooth)

def jacard_coef(y_true, y_pred, smooth = 100.0): #between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    y_inter = K.sum(y_true_f * y_pred_f)
    y_sum = K.sum(y_true_f) + K.sum(y_pred_f)
    
    return (y_inter + smooth) / (y_sum - y_inter + smooth)

def jacard_coef_loss(y_true, y_pred, smooth = 100.0):
    return (1 - jacard_coef(y_true, y_pred)) * smooth

def dice_coef_loss(y_true, y_pred, smooth = 100.0):
    return (1 - dice_coef(y_true, y_pred)) * smooth

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

def result_jaccard_coeff(img1, img2):
    img1_t = K.variable(img1)
    img2_t = K.variable(img2)

    return K.eval(jacard_coef(img1_t, img2_t))
    
if __name__ == '__main__':
    batch_size = 128
    model = us.u_net_segmentation(chanels = 3)
    model.compile(optimizer = Adam(lr = 1e-4), loss = jacard_coef_loss, metrics = ['accuracy', dice_coef, jacard_coef])
    im = np.array(load_transform_pictures('polyps\\input\\data\\*.jpg'))
    mask = np.array(load_transform_pictures('polyps\\input\\label\\*.jpg'))
    model.fit(x = im,
                y=mask,
                batch_size = batch_size,
                epochs = 1, 
                validation_split= 0.2
            )
    
    lab_pred = model.predict(im[:1])

    print("\nJaccard Coefficient for result: " + str(round(result_jaccard_coeff(mask[0], lab_pred[0])*100, 2)) + "%")

    plt.imshow(lab_pred[0])
    plt.show()
