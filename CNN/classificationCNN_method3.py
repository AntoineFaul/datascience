import numpy as np

import pickle
import sys

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import callbacks

import CNN.classificationCNN as classCNN
import polyps.file_manager as fm
import CNN.cnnConfig as cnnConfig

# %% Import data

if __name__ == "__main__":

    if(len(sys.argv) > 1):
        config = cnnConfig.getConfig(sys.argv[1])
    else:
        config = cnnConfig.getConfig("...")

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

    # Display the first images in data
    classCNN.plot_data(train_dataSet, test_dataSet)

# %% Model Creation

    model1 = classCNN.createModel(train_data.shape[1:], nClasses)
    model1.compile(optimizer=config['compile']['optimizer'],
                   loss=config['compile']['loss'], metrics=config['compile']['metrics'])
    model1.summary()

# %% fit the program

    callBacks = callbacks.EarlyStopping(
        monitor=config['callbacks']['monitor'],
        min_delta=config['callbacks']['min_delta'],
        patience=config['callbacks']['patience'],
        verbose=config['callbacks']['verbose'],
        mode=config['callbacks']['mode'],
        baseline=config['callbacks']['baseline'],
        restore_best_weights=config['callbacks']['restore'])

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
            callbacks=[callBacks] if config['fit']['callbacks'] == 'callbacks' else None)
    else:
        print("Fit without a generator")
        history1 = model1.fit(train_data, train_labels_one_hot,
                              batch_size=config['fit']['batch_size'],
                              epochs=config['fit']['epochs'],
                              verbose=config['fit']['verbose'],
                              validation_data=(
                                  validation_data, validation_labels_one_hot),
                              callbacks=[callBacks] if config['fit']['callbacks'] == 'callbacks' else None)

# %%

    eval1 = model1.evaluate(test_data, test_labels_one_hot)

    print(eval1)

    predict = model1.predict(test_data)
    predict_class = model1.predict_classes(test_data)

    A = [predict_class[i] == test_dataSet[1][i] for i in range(len(predict))]
    predict_succes = predict[A, predict_class[A]]
    A = [not i for i in A[:]]
    predict_fails = predict[A, predict_class[A]]

    print(predict_succes)

    succes = (len(predict_succes) /
              (len(predict_succes) + len(predict_fails)) * 100)
    print('Prediction : ' + str(succes) + '%')

    # classCNN.plot_performances( predict_succes=predict_succes, predict_fails=predict_fails)
    # classCNN.plot_history(history1)

    classCNN.plot_fullInformation(
        succes, history1, predict_succes=predict_succes, predict_fails=predict_fails, name=config['nameModel'])

# %%

    # serialize model to JSON
    model_json = model1.to_json()
    with open(fm.make_path(config['folderModel'], config['nameModel'] + ".json"), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model1.save_weights(fm.make_path(
        config['folderModel'], config['nameModel'] + ".h5"))
    print("Saved model to disk")

    # Overwrites any existing file.
    with open(fm.make_path(config['folderModel'], config['nameModel'] + ".pkl"), 'wb') as output:
        pickle.dump(history1, output, pickle.HIGHEST_PROTOCOL)
