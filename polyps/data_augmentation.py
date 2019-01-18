from keras.preprocessing.image import ImageDataGenerator
from file_manager import make_path, clean_folder
import os
import glob
import platform

TARGET_SIZE = (224, 224)
SEED = 1


def createGenerator():
    return  ImageDataGenerator(
            #zoom_range= (0.8,1), # Range for random zoom (<1 : zoom in)
            #width_shift_range=0.01, 
            #height_shift_range=0.01, # shift by fraction of total height
            #rotation_range = 45,  # Degree range for random rotations
            brightness_range = (0.75,1.25),
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True  # randomly flip images
        )

def generator_flow(image_path, newImage_path, batch_size, classData):
    return createGenerator().flow_from_directory(
            image_path,
            batch_size = batch_size,
            classes = [classData],
            target_size = TARGET_SIZE,
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = newImage_path + '\\' + classData,
            seed = SEED
        )

def dataWithLabel_Generator(multiplier, image_path, newImage_path, classData, classLabel):
    clean_folder(newImage_path+PATH_SEP+classData)
    fileList = os.listdir(newImage_path+"\\"+classLabel)

    for fileName in fileList:
        os.remove(newImage_path+"\\"+classLabel+"\\"+fileName)

    path1 = make_path(image_path, classData, "*.jpg")
    path2 = make_path(image_path, classLabel, "*.jpg")
    print("\nAugmentation of classe : " + make_path(image_path, classData))
    print("Augmentation of classe : " + make_path(image_path, classLabel, "*.jpg"))

    imageGenerator = generator_flow(image_path, newImage_path, len(glob.glob(path1)), classData)
    maskGenerator = generator_flow(image_path, newImage_path, len(glob.glob(path2)), classLabel)
    
    trainGenerator = zip(imageGenerator, maskGenerator)
        
    for i,batch in enumerate(trainGenerator):
        if(i >= multiplier-1):
            break

    
if __name__ == "__main__":
    # In the folder input, there is two subfolders, the first with pictures and the second with the masks.
    # Same for output folder
    folderInput = make_path('origin')
    folderOutput = make_path('input')

    picture_multiplier = 4 # Output number = ${picture_multiplier} * Input_number
    classData="data" # name of the subfolder containing the picures
    classLabel="label" # name of the subfolder containing the markers
    
    # This function will generate the pictures/marker
    # /!\ All of files inside the both subfolders of the Output folder will be delete before.
    dataWithLabel_Generator(multiplier = picture_multiplier, image_path = folderInput, newImage_path = folderOutput, classData = classData, classLabel = classLabel)
