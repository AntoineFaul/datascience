from keras.models import model_from_json
from keras.utils import to_categorical

import pickle

import classificationCNN as classCNN

folderTest = "C:\\Users\\Mathieu\\Google Drive\\SDU\\DSC\\Project\\DataAugmented_testSeparated\\Test"
maxsize = (64, 64)

def pickled_items(filename):
    """ Unpickle a file of pickled data. """
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

if __name__ == "__main__":
    
    # Importation testing dataSet
    test_dataSet = classCNN.create_data( folderTest, True, maxsize)
    print("importation of " + str(len(test_dataSet[0])) + " pictures as test data")    

    # Preparation of the arrays
    nRows, nCols, nDims = test_dataSet[0].shape[1:]
    test_data = test_dataSet[0].reshape( test_dataSet[0].shape[0], nRows, nCols, nDims)

    # Change to float datatype
    test_data = test_data.astype('float32')

    # Scale the data to lie between 0 to 1
    test_data /= 255

    # Change the labels from integer to categorical data
    test_labels_one_hot = to_categorical(test_dataSet[1])

    # load json and create model
    json_file = open('modelSave\\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelSave\\model.h5")
    print("Loaded model from disk")
     
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = eval1 = loaded_model.evaluate(test_data, test_labels_one_hot)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    for history in pickled_items('modelSave\\history.pkl'):
        classCNN.plot_history( history)