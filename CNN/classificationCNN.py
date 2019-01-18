# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:02:08 2019

@author: Mathieu
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.utils import shuffle


# %% Function definition

def createModel(input_shape, nClasses):
    model = Sequential()
    model.add(Conv2D(5, (5, 5), padding='same',
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(5, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(5, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(5, activation='softmax'))
    # model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model


def createGenerator():

    generator = ImageDataGenerator(
        # randomly rotate images in the range (degrees, 0 to 180)
        # rotation_range=180,
        # zoom_range=1.,  # Range for random zoom
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    return generator


def load_pictures(folder, maxsize):
    data = []
    count = 0
    for filename in glob.glob(folder):
        im = Image.open(filename)
        im.thumbnail(maxsize)
        im = np.array(im)
        data.append(im)
        count = count + 1
    return(data)


def create_data( folder, sameSize, maxsize):
    count_classes = 0
    min_data = -1
    data = []
    label = []
    print(str(len(os.listdir(folder))) + " classes found")
    for classe in os.listdir(folder):
        print("loading classe : " + folder + "\\" + classe + "\\*.jpg\n")
        data_current = load_pictures(
            folder + "\\" + classe + "\\*.jpg", maxsize)
        label_current = [count_classes for i in range(len(data_current))]
        data.append(data_current)
        label.append(label_current)
        count_classes = count_classes + 1

        if min_data == -1 or min_data > len(data_current):
            min_data = len(data_current)

    dataToReturn = []
    labelToReturn = []

    if sameSize:
        for i in range(len(data)):
            dataToReturn = dataToReturn + data[i][:min_data]
            labelToReturn = labelToReturn + label[i][:min_data]

    return(np.array(dataToReturn), np.array(labelToReturn))


def shuffle_split_data(data, label, perCent_Test):
    x, y = shuffle(data, label)
    maxIndex = int(perCent_Test * len(data))
    x_train = x[:maxIndex]
    x_test = x[maxIndex:]
    y_train = y[:maxIndex]
    y_test = y[maxIndex:]

    return (x_train, y_train), (x_test, y_test)


def plot_data(train_dataSet, test_dataSet):
    f = plt.figure()
    f.suptitle("0 = Clean, 1 = Dirty", fontsize=12)
    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(train_dataSet[0][0], cmap='gray')
    plt.title("Label : {}".format(train_dataSet[1][0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_dataSet[0][0], cmap='gray')
    plt.title("Label : {}".format(test_dataSet[1][0]))


def plot_history( history):
    plt.figure()
    plt.subplot(121)
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.xlabel('Epochs ', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    #plt.figure(figsize=[8, 6])
    plt.subplot(122)
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=10, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Epochs ', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    
def plot_performances( model, history, test_data, test_label):
   
    predict = model.predict( test_data)
    predict_class = model.predict_classes( test_data)
    
    A = [ predict_class[i]==test_label[i] for i in range( len(predict))]
    predict_succes = predict[A, predict_class[A]]
    A = [ not i for i in A[:]]
    predict_fails = predict[A, predict_class[A]]
    
    print( predict_succes)
    
    fig, axs = plt.subplots(1, 2)

    # succes plot
    bp = axs[0].boxplot(predict_succes, patch_artist=True)
    axs[0].set_title('Succes')
    
    # succes plot
    bp2 = axs[1].boxplot(predict_fails, patch_artist=True)
    axs[1].set_title('Fails')
    succes = (len(predict_succes)/(len(predict_succes)+len(predict_fails))*100)
    print('Prediction : ' + str(succes) + '%')
    
    for element in ['boxes']:
        plt.setp(bp[element], color='green')
        plt.setp(bp2[element], color='red')
    
    for patch in bp['boxes']:
        patch.set(facecolor='honeydew')
    for patch in bp2['boxes']:
        patch.set(facecolor='mistyrose')  
        
    return predict,predict_class,predict_succes,predict_fails