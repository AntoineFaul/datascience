from keras.models import Model, load_model
from polyps import data_augmentation, data_transformation, file_manager as fm
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import backend as K
import glob
import model
from config import config


def load_transform_pictures(folder):
    x_train = []

    for filename in glob.glob(folder):
        im = fm.load_image(filename)
        x_train.append(im)

    return(x_train)
    
def create_binary(a, n=1):
    binary = []
    for i in range(len(a)):
    # a : Input array
    # n : We want n-max element position to be set to 1
        out = np.zeros_like(a[i])
        out[np.arange(len(a[i])), np.argpartition(a[i],-n, axis=1)[:,-n]] = 1
        binary.append(out)
    return(binary)

def load_file_name(folder):
    x_train = []
    for filename in glob.glob(folder):
        x_train.append(filename)
    return(x_train)

def pixel_class(c):
    if c == 1:
        return config['color']['rgb']['red']
    elif c == 2:
        return config['color']['rgb']['green']
    elif c == 3:
        return config['color']['rgb']['blue']
    else:
        return config['color']['rgb']['black']
        
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

def write_image(array, directory, file_name):
    fm.clean_folder(fm.make_path('polyps', 'output','data'))
    index = 0
    img_store = []

    for image in array:
        img = Image.new('RGB', config['image_size'], "white")

        for i in range(config['image_max']):
            for j in range(config['image_max']):
                img.putpixel((i, j), image[i][j])
        n = (file_name[index].split(config['path_sep']))[-1]
        name = '{}'.format(n)
        img.save(fm.make_path(directory, name))
        img_store.append(np.array(img))
        index = index+1
    return img_store

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred): # between 0 and 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def result_jaccard_coeff(img1, img2):
    img1_t = K.variable(img1)
    img2_t = K.variable(img2)
    return K.eval(jacard_coef(img1_t, img2_t))
 
def contingency_table(lab_pred,mask_test):
    fp,tp,tn,fn,sum_bg,sum_po,sum_wa,sum_di=(0,0,0,0,0,0,0,0)
    classes = 4
    cont_store = []
    for j in range(len(test)):
        for i in range(classes):
            output_binary = np.array(create_binary(lab_pred[j]))*2
            mask_output_comparison = np.subtract(mask_test[j][:,:,i],output_binary[:,:,i])
            fp = (mask_output_comparison==-2).sum()
            tp =(mask_output_comparison==-1).sum()
            tn =(mask_output_comparison==0).sum()
            fn =(mask_output_comparison==1).sum()
            cont_store.append((fp,tp,tn,fn))
    background,polype,wall,dirt = ([],[],[],[])
    for j in range(classes):
        for i in range(len(test)):
            sum_bg += cont_store[i*4][j]
            sum_po += cont_store[i*4+1][j]
            sum_wa += cont_store[i*4+2][j]
            sum_di += cont_store[i*4+3][j]
        background.append(sum_bg)
        polype.append(sum_po)
        wall.append(sum_wa)
        dirt.append(sum_di)
        sum_bg,sum_po,sum_wa,sum_di=(0,0,0,0)
    background=background/sum(background)
    polype =polype/sum(polype)
    wall=wall/sum(wall)
    dirt=dirt/sum(dirt)

    print("Background: "+"FP: " +str(round(background[0],3))+" TP: "+ str(round(background[1],3)) + " TN: " +str(round(background[2],3))+" FN: " +str(round(background[3],3)))
    print("Polype: "+"FP: " +str(round(polype[0],3))+" TP: "+ str(round(polype[1],3)) + " TN: " +str(round(polype[2],3))+" FN: " +str(round(polype[3],3)))
    print("Wall: "+"FP: " +str(round(wall[0],3))+" TP: "+ str(round(wall[1],3)) + " TN: " +str(round(wall[2],3))+" FN: " +str(round(wall[3],3)))
    print("Dirt: "+"FP: " +str(round(dirt[0],3))+" TP: "+ str(round(dirt[1],3)) + " TN: " +str(round(dirt[2],3))+" FN: " +str(round(dirt[3],3)))

def overall_contingency_table(lab_pred,mask_test):
    fp,tp,tn,fn =(0,0,0,0)
    for j in range(len(test)):
        output_binary = np.array(create_binary(lab_pred[j]))*2
        mask_output_comparison = np.subtract(mask_test[j],output_binary)
        fp += (mask_output_comparison==-2).sum()
        tp +=(mask_output_comparison==-1).sum()
        tn +=(mask_output_comparison==0).sum()
        fn +=(mask_output_comparison==1).sum()
    sum_classses = (fp+tp+tn+fn)
    fp = fp/ sum_classses
    tp = tp/sum_classses
    tn = tn/sum_classses
    fn = fn/sum_classses
    print("FP: " +str(round(fp,3))+" TP: "+ str(round(tp,3)) + " TN: " +str(round(tn,3))+" FN: " +str(round(fn,3)))


if __name__ == "__main__":
    data_augmentation.execute()

    batch_size = config['batch_size']
#    model = model.u_net(IMG_SIZE = config['image_dimension']) #what does the Adam optimizer do
    model = model.u_net_batch_norm_upc()
    model.compile(optimizer = Adam(lr = 1e-4), loss = config['loss'] , metrics = config['metrics'])


    img_train = np.array(load_transform_pictures(fm.make_path('polyps', 'training', 'data', '*.jpg')))
    img_val = np.array(load_transform_pictures(fm.make_path('polyps', 'validation', 'data', '*.jpg')))
    img_test = np.array(load_transform_pictures(fm.make_path('polyps', 'test','data', '*.jpg')))

    test_name = np.array(load_file_name(fm.make_path('polyps', 'test', 'data', '*.jpg')))

    output = fm.make_path('polyps', 'output','data')

    mask_train = np.array(data_transformation.create_binary_masks(path = fm.make_path('polyps', 'training', 'label')),dtype = "float32") 
    mask_val = np.array(data_transformation.create_binary_masks(path = fm.make_path('polyps', 'validation', 'label')),dtype = "float32") 
    mask_test = np.array(data_transformation.create_binary_masks(path = fm.make_path('polyps', 'test', 'label')),dtype = "float32")
  
    history = model.fit(x = img_train, y = mask_train,
                        #validation_split = config['validation_split'],
                        validation_data = (img_val, mask_val),
                        steps_per_epoch = config['fit']['steps_per_epoch'],
                        validation_steps = config['fit']['validation_steps'],
                        epochs = config['fit']['epochs'],
                        shuffle = config['fit']['shuffle'],
                        batch_size = config['fit']['batch_size'],
                        class_weight = config['fit']['class_weight'],
                        callbacks = config['fit']['callbacks']  
                    )

    history = history.history
    lab_pred = model.predict(img_test, verbose = 1)
#    evaluate = model.evaluate(x = test, y = mask_test, batch_size = batch_size)
    display_im = write_image(merge(lab_pred), output, test_name)
    plt.imshow(display_im[0])#plots the first picture
    plt.show()
    print("Evaluation : Loss: "+ str(evaluate[0]) + ", Accuracy: " + str(evaluate[1]) + ", Dice Coefficient: " + str(evaluate[2]) + ", Jacard Coefficient: " + str(evaluate[3]))
    
    overall_contingency_table(lab_pred,mask_test)
    contingency_table(lab_pred,mask_test)
    
