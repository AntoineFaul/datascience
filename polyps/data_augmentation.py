from keras.preprocessing.image import ImageDataGenerator
from polyps.file_manager import make_path, clean_folder_group, remove_except_files
from random  import choice
import glob
import os
from config import config

INPUT_PATH = make_path('polyps', 'origin')
OUTPUT_PATH = make_path('polyps', 'input')
TEST_PATH = make_path('polyps', 'test')

CLASS_DATA = 'data'
CLASS_LABEL = 'label'


def createGenerator():
    return  ImageDataGenerator(
            zoom_range = config['augmentation']['zoom_range'], # range for random zoom (<1 : zoom in)
            width_shift_range = config['augmentation']['width_shift_range'], # shift by fraction of total width
            height_shift_range = config['augmentation']['height_shift_range'], # shift by fraction of total height
            rotation_range = config['augmentation']['rotation_range'], # degree range for random rotations
            brightness_range = config['augmentation']['brightness_range'],
            horizontal_flip = config['augmentation']['flip']['horizontal'], # randomly flip images
            vertical_flip = config['augmentation']['flip']['vertical']  # randomly flip images
        )

def generator_flow(image_path, batch_size, class_type):
    return createGenerator().flow_from_directory(
            image_path,
            batch_size = batch_size,
            classes = [class_type],
            target_size = config['image_size'],
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = make_path(OUTPUT_PATH , class_type),
            seed = config['seed']
        )

def dataWithLabel_Generator():
    clean_folder_group(OUTPUT_PATH, CLASS_DATA, CLASS_LABEL)

    path1 = make_path(INPUT_PATH, CLASS_DATA, "*.jpg")
    path2 = make_path(INPUT_PATH, CLASS_LABEL, "*.jpg")
    print("\nAugmentation of classe : " + make_path(INPUT_PATH, CLASS_DATA))
    print("Augmentation of classe : " + make_path(INPUT_PATH, CLASS_LABEL))

    imageGenerator = generator_flow(INPUT_PATH, len(glob.glob(path1)), CLASS_DATA)
    maskGenerator = generator_flow(INPUT_PATH, len(glob.glob(path2)), CLASS_LABEL)
    
    trainGenerator = zip(imageGenerator, maskGenerator)
        
    for i, batch in enumerate(trainGenerator):
        if (i >= config['multiplier']-1):
            break

def extract_data_test():
    clean_folder_group(TEST_PATH, CLASS_DATA, CLASS_LABEL)
    images = os.listdir(make_path(OUTPUT_PATH, CLASS_DATA))
    remove_except_files(images)
    
    im_list = []

    for i in range(int(len(images)*config['test_split'])):
        im = choice(images)
        im_list.append(im)
        images.remove(im)

    for filename in im_list:
        os.rename(make_path(OUTPUT_PATH, CLASS_DATA, filename), make_path(TEST_PATH, CLASS_DATA, filename))
        os.rename(make_path(OUTPUT_PATH, CLASS_LABEL, filename), make_path(TEST_PATH, CLASS_LABEL, filename))

def execute():
    # This function will generate the pictures/marker
    # /!\ All of files inside the both subfolders of the Output folder will be delete before.
    dataWithLabel_Generator()
    extract_data_test()

    print("Augmentation done.\n")
