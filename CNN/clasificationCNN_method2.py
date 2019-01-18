from classificationCNN import *

from keras.utils import to_categorical

#%%

#folder = "C:\\Users\\Mathieu\\Google Drive\\SDU\\DSC\\Project\\Data"
folderTest = "C:\\Users\\Mathieu\\Google Drive\\SDU\\DSC\\Project\\DataAugmented_testSeparated\\Test"
folderTrain = "C:\\Users\\Mathieu\\Google Drive\\SDU\\DSC\\Project\\DataAugmented_testSeparated\\Train"
maxsize = (64, 64)
perCent_Test = 0.8
perCent_Validation = 0.8
batch_size = 224
epochs = 50

#%% Import data

if __name__ == "__main__":

    (data, label) = create_data( folderTrain, True, maxsize)
    print(len(data))
    print(" ")
    print(len(label))
    
    (train_dataSet, validation_dataSet) = shuffle_split_data(data, label, perCent_Validation)
    
    test_dataSet = create_data( folderTest, True, maxsize)
    print(len(data))
    print(" ")
    print(len(label))

#%% preprocessing

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
    classes = np.unique(label)
    nClasses = len(classes)

    # Display the first images in data
    plot_data(train_dataSet, test_dataSet)

#%% Model Creation

    #train_data(train_data=train_data, train_label=train_labels_one_hot,               validation_data=test_data, validation_label=test_labels_one_hot)

    model1 = createModel(train_data.shape[1:], nClasses)
    model1.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model1.summary()
    
    #%%

    history1 = model1.fit(train_data, train_labels_one_hot,
                        batch_size=batch_size, epochs=epochs,
                        verbose=1,
                        validation_data=( validation_data, validation_labels_one_hot))

#%%   
    
    eval1 = model1.evaluate(test_data, test_labels_one_hot)

    print( eval1)
       
    (predict1, predict1_class, predict1_succes, predict1_fails) = plot_performances( model1, history1, test_data, test_dataSet[1])
    plot_history( history1)
        

    