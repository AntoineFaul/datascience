from keras.preprocessing.image import ImageDataGenerator
import os
import glob

def createGenerator():
    
    return  ImageDataGenerator(
            #rotation_range=45, # Degree range for random rotations
            zoom_range= (0.8,1), # Range for random zoom (<1 : zoom in)
            width_shift_range=0.01, 
            height_shift_range=0.01, # shift by fraction of total height
            brightness_range = (0.75,1.25),
            rotation_range = 45,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True)  # randomly flip images

def data_Generator( multiplier, image_path, newImage_path, target_size):

    generator = createGenerator()        
    
    for classe in os.listdir(image_path):
        
        fileList = os.listdir(newImage_path+"\\"+classe)
        for fileName in fileList:
            os.remove(newImage_path+"\\"+classe+"\\"+fileName)

        path = image_path+"\\"+classe+"\\*.jpg"
        print("Augmentation of classe : " + image_path + "\\" + classe + "\n")

        imageGenerator = generator.flow_from_directory(
            image_path,
            batch_size = len(  glob.glob( path)),
            classes = [classe],
            #target_size = target_size,
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = newImage_path + '\\' + classe)
        
        for i,batch in enumerate(imageGenerator):
            print( i)
            if(i >= multiplier-1):
                break

def dataWithLabel_Generator( multiplier, image_path, newImage_path, classData, classLabel, target_size, seed=1):

    generatorData = createGenerator()        
    generatorLabel = createGenerator()        
            
    fileList = os.listdir(newImage_path+"\\"+classData)
    for fileName in fileList:
        os.remove(newImage_path+"\\"+classData+"\\"+fileName)
        
    fileList = os.listdir(newImage_path+"\\"+classLabel)
    for fileName in fileList:
        os.remove(newImage_path+"\\"+classLabel+"\\"+fileName)

    path1 = image_path+"\\"+classData+"\\*.jpg"
    path2 = image_path+"\\"+classLabel+"\\*.jpg"
    print("Augmentation of classe : " + image_path + "\\" + classData + "\n")
    print("Augmentation of classe : " + image_path + "\\" + classLabel + "\n")

    imageGenerator = generatorData.flow_from_directory(
            image_path,
            batch_size = len(  glob.glob( path1)),
            classes = [classData],
            #target_size = target_size,
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = newImage_path + '\\' + classData,
            seed = seed)
    maskGenerator = generatorLabel.flow_from_directory(
            image_path,
            batch_size = len(  glob.glob( path2)),
            classes = [classLabel],
            #target_size = target_size,
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = newImage_path + '\\' + classLabel,
            seed = seed)
    
    trainGenerator = zip(imageGenerator, maskGenerator)
        
    for i,batch in enumerate(trainGenerator):
        print( i)
        if(i >= multiplier-1):
            break
        
    
if __name__ == "__main__":

    # In the folder input, there is two subfolders, the first with pictures and the second with the masks.
    # Same for output folder
        
    folderInput = "C:\\Users\\Mathieu\\Google Drive\\SDU\\DSC\\Project\\Data Science\\Polyps_with_label\\Mathieu"
    folderOutput = "C:\\Users\\Mathieu\\Google Drive\\SDU\\DSC\\Project\\DataPlop"
    picture_multiplier = 1 # Output number = ${picture_multiplier} * Input_number
    classData="Data" # name of the subfolder containing the picures
    classLabel="Label" # name of the subfolder containing the markers
    
    # This function will generate the pictures/marker
    # /!\ All of files inside the both subfolders of the Output folder will be delete before.
    dataWithLabel_Generator(multiplier=picture_multiplier, image_path=folderInput, newImage_path=folderOutput, classData=classData, classLabel=classLabel, target_size= (227,227))