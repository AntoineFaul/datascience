import numpy as np

import pickle
import sys
import os
import matplotlib.pyplot as plt

from keras.utils import to_categorical, print_summary
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import callbacks

from . import classificationCNN as classCNN
from data import manager as fm
from . import cnnConfig as cnnConfig

# %% Import data


def execute(argv):

    config = cnnConfig.getConfig(argv)

    print("Configuration loaded : " + config['nameModel'])
    print("Hit a key to continue ...")
    input()

    # Importation trainning dataSet
    source_dataSet = classCNN.create_data(
        config['folderTrain'], True, config['maxsize'])
    print("importation of " +
          str(len(source_dataSet[0])) + " pictures as training data")

    # Importation testing dataSet
    test_dataSet = classCNN.create_data(
        config['folderTest'], True, config['maxsize'])
    print("importation of " +
          str(len(test_dataSet[0])) + " pictures as test data")

    # Creation of the validation dataSet
    (train_dataSet, validation_dataSet) = classCNN.shuffle_fairSplit_data(
        data=source_dataSet[0], label=source_dataSet[1], perCent_Test=config['perCent_Validation'])
    print("Split the trainning data set")
    print("Train dataSet (" + str(len(train_dataSet[1])) + " pictures) : " + str(len(
        [train_dataSet[1][i] for i in range(len(train_dataSet[1])) if train_dataSet[1][i] == 0])) + " Clean")
    print("Validation dataSet (" + str(len(validation_dataSet[1])) + " pictures) : " + str(len(
        [validation_dataSet[1][i] for i in range(len(validation_dataSet[1])) if validation_dataSet[1][i] == 0])) + " Clean")

# %% preprocessing

    # Preparation of the arrays
    nRows, nCols, nDims = train_dataSet[0].shape[1:]
    train_data = train_dataSet[0].reshape(
        train_dataSet[0].shape[0], nRows, nCols, nDims)
    test_data = test_dataSet[0].reshape(
        test_dataSet[0].shape[0], nRows, nCols, nDims)
    validation_data = validation_dataSet[0].reshape(
        validation_dataSet[0].shape[0], nRows, nCols, nDims)

    # Change to float datatype
    train_data = train_data.astype('float32')
    validation_data = validation_data.astype('float32')
    test_data = test_data.astype('float32')

    # Scale the data to lie between 0 to 1
    train_data /= 255
    validation_data /= 255
    test_data /= 255

    # Change the labels from integer to categorical data
    train_labels_one_hot = to_categorical(train_dataSet[1])
    validation_labels_one_hot = to_categorical(validation_dataSet[1])
    test_labels_one_hot = to_categorical(test_dataSet[1])

    # Find the unique numbers from the train labels
    classes = np.unique(train_dataSet[1])
    nClasses = len(classes)

    print(str(nClasses) + " classes detected")

    # Display the first images in data
    classCNN.plot_data(train_dataSet, test_dataSet)

# %% Model Creation

    model1 = classCNN.createModel(
        train_data.shape[1:], nClasses, config['model'])
    model1.compile(optimizer=config['compile']['optimizer'],
                   loss=config['compile']['loss'], metrics=config['compile']['metrics'])
    model1.summary()

    # Open the file
    with open(fm.make_path(config['folderModel'], config['nameModel'], config['nameModel'] + '.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model1.summary(print_fn=lambda x: fh.write(x + '\n'))

# %% fit the program

    print("Hit a key to train the model ...")
    input()

    callBacks1 = callbacks.EarlyStopping(
        monitor=config['callbacks1']['monitor'],
        min_delta=config['callbacks1']['min_delta'],
        patience=config['callbacks1']['patience'],
        verbose=config['callbacks1']['verbose'],
        mode=config['callbacks1']['mode'],
        baseline=config['callbacks1']['baseline'],
        restore_best_weights=config['callbacks1']['restore'])

    callBacks2 = callbacks.EarlyStopping(
        monitor=config['callbacks2']['monitor'],
        min_delta=config['callbacks2']['min_delta'],
        patience=config['callbacks2']['patience'],
        verbose=config['callbacks2']['verbose'],
        mode=config['callbacks2']['mode'],
        baseline=config['callbacks2']['baseline'],
        restore_best_weights=config['callbacks2']['restore'])

    if(config['fit']['callbacks'] == 1):
        print(callBacks1.monitor)

    generator = ImageDataGenerator(
        rotation_range=config['generator']['rotation'],
        zoom_range=config['generator']['zoom'],
        width_shift_range=config['generator']['width_shift'],
        height_shift_range=config['generator']['height_shift'],
        brightness_range=config['generator']['brightness'],
        horizontal_flip=config['generator']['horizontal_flip'],
        vertical_flip=config['generator']['vertical_flip'])

    if(config['fit']['generator'] == 'generator'):
        print("Fit with a generator")
        history1 = model1.fit_generator(
            generator.flow(train_data,
                           train_labels_one_hot, batch_size=config['fit']['batch_size']),
            steps_per_epoch=int(
                np.ceil(train_data.shape[0] / float(config['fit']['batch_size']))),
            epochs=config['fit']['epochs'],
            verbose=config['fit']['verbose'],
            validation_data=(
                validation_data, validation_labels_one_hot),
            workers=4,
            shuffle=config['fit']['shuffle'],
            callbacks=[callBacks1] if config['fit']['callbacks'] == 1 else [callBacks1, callBacks2] if config['fit']['callbacks'] == 2 else None)
    else:
        print("Fit without a generator")
        history1 = model1.fit(train_data, train_labels_one_hot,
                              batch_size=config['fit']['batch_size'],
                              epochs=config['fit']['epochs'],
                              verbose=config['fit']['verbose'],
                              shuffle=config['fit']['shuffle'],
                              validation_data=(
                                  validation_data, validation_labels_one_hot),
                              callbacks=[callBacks1] if config['fit']['callbacks'] == 1 else [callBacks1, callBacks2] if config['fit']['callbacks'] == 2 else None)

# %%

    eval1 = model1.evaluate(test_data, test_labels_one_hot)

    print(eval1)

    predict = model1.predict(test_data)
    predict_class = model1.predict_classes(test_data)

    A = [predict_class[i] == test_dataSet[1][i] for i in range(len(predict))]
    predict_succes = predict[A, predict_class[A]]
    A = [not i for i in A[:]]
    predict_fails = predict[A, predict_class[A]]

    succes = (len(predict_succes) /
              (len(predict_succes) + len(predict_fails)) * 100)
    print('Prediction : ' + str(succes) + '%')

    # Analysis of the true and false result
    trueClean = len(predict_class[[predict_class[i] == test_dataSet[1]
                                   [i] and test_dataSet[1][i] == 0 for i in range(len(predict))]])
    trueDirty = len(predict_class[[predict_class[i] == test_dataSet[1]
                                   [i] and test_dataSet[1][i] == 1 for i in range(len(predict))]])
    falseClean = len(predict_class[[predict_class[i] != test_dataSet[1]
                                    [i] and test_dataSet[1][i] == 0 for i in range(len(predict))]])
    falseDirty = len(predict_class[[predict_class[i] != test_dataSet[1]
                                    [i] and test_dataSet[1][i] == 1 for i in range(len(predict))]])

    # classCNN.plot_performances( predict_succes=predict_succes, predict_fails=predict_fails)
    # classCNN.plot_history(history1)

    classCNN.plot_fullInformation(
        succes, history1, predict_succes=predict_succes, predict_fails=predict_fails, path=config['folderModel'], name=config['nameModel'], analysis=[trueClean, falseClean, trueDirty, falseDirty])


# %% uncomment to save model
#
#   # folder preparation
#   if not os.path.exists(fm.make_path(config['folderModel'], config['nameModel'])):
#        os.makedirs(fm.make_path(config['folderModel'], config['nameModel']))
#
#    # serialize model to JSON
#    model_json = model1.to_json()
#    with open(fm.make_path(config['folderModel'], config['nameModel'], config['nameModel'] + ".json"), "w") as json_file:
#        json_file.write(model_json)
#
#    # serialize weights to HDF5
#    model1.save_weights(fm.make_path(
#        config['folderModel'], config['nameModel'], config['nameModel'] + ".h5"))
#    print("Saved model to disk")

#    # save the history model
#    with open(fm.make_path(config['folderModel'], config['nameModel'], config['nameModel'] + ".pkl"), 'wb') as output:
#        pickle.dump(history1, output, pickle.HIGHEST_PROTOCOL)

    # save the plot of the full informations
#    fig = plt.figure(3)
#    fig.savefig(fm.make_path(
#        config['folderModel'], config['nameModel'], config['nameModel'] + ".png"))
