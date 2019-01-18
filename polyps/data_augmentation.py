from keras.preprocessing.image import ImageDataGenerator
from polyps.file_manager import make_path, clean_folder_group
from random  import choice
import glob
import os

TEST_PERC = 0.2
TARGET_SIZE = (224, 224)
SEED = 1

input_path = make_path('polyps', 'origin')
output_path = make_path('polyps', 'input')
test_path = make_path('polyps', 'test')

CLASS_DATA = 'data'
CLASS_LABEL = 'label'


def createGenerator():
    return  ImageDataGenerator(
            #zoom_range= (0.8,1), # Range for random zoom (<1 : zoom in)
            #width_shift_range=0.01, 
            #height_shift_range=0.01, # shift by fraction of total height
            #rotation_range = 45,  # Degree range for random rotations
            brightness_range = (0.75,1.25),
            horizontal_flip = True,  # randomly flip images
            vertical_flip = True  # randomly flip images
        )

def generator_flow(image_path, batch_size, class_type):
    return createGenerator().flow_from_directory(
            image_path,
            batch_size = batch_size,
            classes = [class_type],
            target_size = TARGET_SIZE,
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = make_path(output_path , class_type),
            seed = SEED
        )

def dataWithLabel_Generator(multiplier):
    clean_folder_group(output_path, CLASS_DATA, CLASS_LABEL)

    path1 = make_path(input_path, CLASS_DATA, "*.jpg")
    path2 = make_path(input_path, CLASS_LABEL, "*.jpg")
    print("\nAugmentation of classe : " + make_path(input_path, CLASS_DATA))
    print("Augmentation of classe : " + make_path(input_path, CLASS_LABEL))

    imageGenerator = generator_flow(input_path, len(glob.glob(path1)), CLASS_DATA)
    maskGenerator = generator_flow(input_path, len(glob.glob(path2)), CLASS_LABEL)
    
    trainGenerator = zip(imageGenerator, maskGenerator)
        
    for i,batch in enumerate(trainGenerator):
        if (i >= multiplier-1):
            break

def extract_data_test():
    clean_folder_group(test_path, CLASS_DATA, CLASS_LABEL)
    images = os.listdir(make_path(output_path, CLASS_DATA))
    im_list = []

    images.remove('.gitkeep')

    for i in range(int(len(images)*TEST_PERC)):
        im = choice(images)
        im_list.append(im)
        images.remove(im)

    for filename in im_list:
        os.rename(make_path(output_path, CLASS_DATA, filename), make_path(test_path, CLASS_DATA, filename))
        os.rename(make_path(output_path, CLASS_LABEL, filename), make_path(test_path, CLASS_LABEL, filename))

def execute():
    picture_multiplier = 1 # Output number = ${picture_multiplier} * Input_number
    classData = 'data' # name of the subfolder containing the picures
    classLabel = 'label' # name of the subfolder containing the markers
    
    # This function will generate the pictures/marker
    # /!\ All of files inside the both subfolders of the Output folder will be delete before.
    dataWithLabel_Generator(picture_multiplier)
    extract_data_test()

    print("Augmentation done.\n")
