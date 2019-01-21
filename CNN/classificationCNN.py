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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import polyps.file_manager as fm

# %% Function definition


def createModel(input_shape, nClasses, modelDict):
    model = Sequential()

    for line in modelDict:
        if(line['index'] == 'Conv2D'):
            if(line['shape']):
                model.add(Conv2D(line['filters'],
                                 line['kernel_size'],
                                 padding=line['padding'],
                                 activation=line['activation'],
                                 input_shape=input_shape))
            else:
                model.add(Conv2D(line['filters'],
                                 line['kernel_size'],
                                 padding=line['padding'],
                                 activation=line['activation']))
        elif(line['index'] == 'MaxPooling2D'):
            model.add(MaxPooling2D(pool_size=line['pool_size']))
        elif(line['index'] == 'Dropout'):
            model.add(Dropout(line['rate']))
        elif(line['index'] == 'Flatten'):
            model.add(Flatten())
        elif(line['index'] == 'Dense'):
            if(line['units'] == 'nClasses'):
                model.add(Dense(units=nClasses,
                                activation=line['activation']))
            else:
                model.add(Dense(units=line['units'],
                                activation=line['activation']))
        else:
            raise ValueError('Model can\'t be trainned')

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


def create_data(folder, sameSize, maxsize):
    count_classes = 0
    min_data = -1
    data = []
    label = []
    print(str(len(os.listdir(folder))) + " classes found")
    for classe in os.listdir(folder):
        print(classe)
        if(classe == ".gitkeep"):
            continue
        print("loading classe : " + fm.make_path(folder, classe, "*.jpg") + "\n")
        data_current = load_pictures(
            fm.make_path(folder, classe, "*.jpg"), maxsize)
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
    maxIndex = int((1 - perCent_Test) * len(data))
    x_train = x[:maxIndex]
    x_test = x[maxIndex:]
    y_train = y[:maxIndex]
    y_test = y[maxIndex:]

    return (x_train, y_train), (x_test, y_test)


def shuffle_fairSplit_data(data, label, perCent_Test):

    maxIndex = int((1 - perCent_Test) * len(data))
    rows, cols, channels = data.shape[1:]
    x_train = np.empty(shape=(maxIndex, rows, cols, channels), dtype=np.uint8)
    x_test = np.empty(shape=(len(data) - maxIndex, rows,
                             cols, channels), dtype=np.uint8)
    y_train = np.empty(shape=(maxIndex), dtype=np.uint8)
    y_test = np.empty(shape=(len(data) - maxIndex), dtype=np.uint8)

    cptClean = 0
    cptDirty = 0
    cptTrain = 0
    cptTest = 0

    (data, label) = shuffle(data, label)

    for i in range(len(data)):
        if(label[i] == 0):  # Clean
            if(cptClean < int(maxIndex / 2)):
                x_train[cptTrain] = data[i]
                y_train[cptTrain] = 0
                cptTrain += 1
            else:
                x_test[cptTest] = data[i]
                y_test[cptTest] = 0
                cptTest += 1
            cptClean += 1
        else:
            if(cptDirty < int(maxIndex / 2)):
                x_train[cptTrain] = data[i]
                y_train[cptTrain] = 1
                cptTrain += 1
            else:
                x_test[cptTest] = data[i]
                y_test[cptTest] = 1
                cptTest += 1
            cptDirty += 1

    return (x_train, y_train), (x_test, y_test)


def plot_data(train_dataSet, test_dataSet):
    f = plt.figure(1)
    f.clear()
    f.suptitle("0 = Clean, 1 = Dirty", fontsize=12)
    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(train_dataSet[0][0], cmap='gray')
    plt.title("Label : {}".format(train_dataSet[1][0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_dataSet[0][0], cmap='gray')
    plt.title("Label : {}".format(test_dataSet[1][0]))

    f.show()


def plot_history(history):
    plt.figure()
    plt.subplot(121)
    plt.plot(history.history['loss'], 'r-+', linewidth=1.5)
    plt.plot(history.history['val_loss'], 'b-+', linewidth=1.5)
    plt.xlabel('Epochs ', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # plt.figure(figsize=[8, 6])
    plt.subplot(122)
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=10,
               bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
    plt.xlabel('Epochs ', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)


def plot_performances(predict_succes, predict_fails):

    fig, axs = plt.subplots(1, 2)

    # succes plot
    bp = axs[0].boxplot(predict_succes, patch_artist=True)
    axs[0].set_title('Succes')

    # succes plot
    bp2 = axs[1].boxplot(predict_fails, patch_artist=True)
    axs[1].set_title('Fails')

    for element in ['boxes']:
        plt.setp(bp[element], color='green')
        plt.setp(bp2[element], color='red')

    for patch in bp['boxes']:
        patch.set(facecolor='honeydew')
    for patch in bp2['boxes']:
        patch.set(facecolor='mistyrose')


def plot_fullInformation(predictPerCent, history, predict_succes, predict_fails, path, name, analysis):

    if(len(analysis) < 4):
        raise ValueError('plot_fullInformation : no enough analysis (<4)')

    nbPicturesTested = analysis[0] + analysis[1] + analysis[2] + analysis[3]
    cells = [["Class :", "Clean", "Dirty"], ["True", str(analysis[0]), str(analysis[2])],
             ["False", str(analysis[1]), str(analysis[3])]]

    fig = plt.figure(3)
    fig.clear()
    gs = GridSpec(2, 2)
    gs.update(hspace=0.4, bottom=0.05)
    fig.suptitle('Model name : ' + str(name), fontsize=18)

    gs1 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, :])
    ax1 = plt.subplot(gs1[0])
    ax1.plot(history.history['loss'], 'r-+', linewidth=1)
    ax1.plot(history.history['val_loss'], 'b-+', linewidth=1)
    ax1.legend(['Training Accuracy', 'Validation Accuracy'],
               fontsize=10, loc='upper right', borderaxespad=0.)
    ax1.grid(True)
    ax1.set_xlabel('Epochs ')
    ax1.set_title('Loss Curves', fontsize=12)

    ax2 = plt.subplot(gs1[1])
    ax2.plot(history.history['acc'], 'r-+', linewidth=1)
    ax2.plot(history.history['val_acc'], 'b-+', linewidth=1)
    ax2.grid(True)
    ax2.set_xlabel('Epochs ')
    ax2.set_title('Accuracy Curves', fontsize=12)

    # box plot part
    gs2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 0], wspace=0.3)

    # succes plot
    ax3 = plt.subplot(gs2[:, 0])
    bp = ax3.boxplot(predict_succes, patch_artist=True)
    ax3.axis([0.75, 1.25, 0.4, 1])
    ax3.set_title(
        'Succes (' + str(round(predictPerCent, 2)) + '%)', fontsize=12)
    ax3.grid(True)
    ax3.set_ylabel('Probability given by the prediction')

    # fails plot
    ax4 = plt.subplot(gs2[:, 1])
    bp2 = ax4.boxplot(predict_fails, patch_artist=True)
    ax4.axis([0.75, 1.25, 0.4, 1])
    ax4.set_title(
        'Fails (' + str(round(100 - predictPerCent, 2)) + '%)', fontsize=12)
    ax4.grid(True)

    for element in ['boxes']:
        plt.setp(bp[element], color='green')
        plt.setp(bp2[element], color='red')

    for patch in bp['boxes']:
        patch.set(facecolor='honeydew')
    for patch in bp2['boxes']:
        patch.set(facecolor='mistyrose')

    # test and table
    gs3 = GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1, 1])

    # print the table
    ax5 = plt.subplot(gs3[0, 0:2])
    ax5.axis('off')
    color = ['lightgreen', 'mistyrose']
    colorCell = [['white', 'white', 'white'],
                 ['tab:green', 'lightgreen', 'lightgreen'],
                 ['tab:red', 'mistyrose', 'mistyrose']]
    table = ax5.table(cellText=cells, loc='center',
                      cellColours=colorCell)
    ax5.set_title('Analysis of the prediction', fontsize=12)
    ax5.text(0.5, -0.05, '(' + str(nbPicturesTested) + ' pictures)',
             horizontalalignment='center', fontsize=12)

    for i in range(3):
        for u in range(3):
            table.get_celld()[(u, i)]._loc = 'center'
            table.get_celld()[(u, i)].set_width = 2
            table.get_celld()[(u, i)].set_height = 2

    # Reading of the summary file
    summary = ""
    with open(fm.make_path(path, name + '.txt'), 'r') as myfile:
        summary += myfile.read().replace("=", "*")

    ax6 = plt.subplot(gs3[:, 2])
    ax6.axis('off')
    ax6.text(0, 1, summary, fontsize=5, verticalalignment='top')
    ax6.set_title('Model summary', fontsize=12,
                  horizontalalignment='left')
