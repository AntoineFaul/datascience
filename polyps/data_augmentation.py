from keras.preprocessing.image import ImageDataGenerator
from polyps.file_manager import make_path, clean_subfolders, remove_except_files, load_image
from random  import choice
import glob
import os
from config import config

INPUT_PATH = make_path('polyps', 'origin')
OUTPUT_PATH = make_path('polyps', 'input')
TEST_PATH = make_path('polyps', 'test')

SUBFOLDERS = os.listdir(INPUT_PATH)


def createGenerator():
    return  ImageDataGenerator(
            zoom_range = config['augmentation']['zoom_range'], # range for random zoom (<1 : zoom in)
            width_shift_range = config['augmentation']['width_shift_range'], # shift by fraction of total width
            height_shift_range = config['augmentation']['height_shift_range'], # shift by fraction of total height
            rotation_range = config['augmentation']['rotation_range'], # degree range for random rotations
            brightness_range = config['augmentation']['brightness_range'], # degree range for brightness
            horizontal_flip = config['augmentation']['flip']['horizontal'], # randomly flip images
            vertical_flip = config['augmentation']['flip']['vertical']  # randomly flip images
        )

def generator_flow(folder, batch_size, subfolder):
    return createGenerator().flow_from_directory(
            directory = folder,
            batch_size = batch_size,
            target_size = config['image_size'],
            classes = [subfolder],
            class_mode = None,
            save_format = 'jpg',
            save_to_dir = make_path(OUTPUT_PATH, subfolder),
            seed = config['seed']
        )

def dataWithLabel_Generator():
    clean_subfolders(OUTPUT_PATH)

    batch_size = len(glob.glob(make_path(INPUT_PATH, SUBFOLDERS[0], '*.jpg')))
    generators = []

    for subfolder in SUBFOLDERS:
        print("\nAugmentation of subfolder: " + subfolder)
        generators.append(generator_flow(INPUT_PATH, batch_size, subfolder))
        print("Creation of " + str(batch_size*config['multiplier']) + " augmented images.")
    
    trainGenerator = zip(generators[0], generators[1])
        
    for i, batch in enumerate(trainGenerator):
        if (i >= config['multiplier']-1):
            break

def test_split():
    input_list = os.listdir(make_path(INPUT_PATH, SUBFOLDERS[0]))
    remove_except_files(input_list)

    test_list = []

    for i in range(int(len(input_list)*config['test_split'])):
        im = choice(input_list)
        test_list.append(im)
        input_list.remove(im)

    test_list = ['_' + filename.split('.jpg')[0] + '_*.jpg' for filename in test_list]

    return test_list

def extract_test(test_list):
    clean_subfolders(TEST_PATH)

    for elt in test_list:
        for subfolder in SUBFOLDERS:
            for filename in glob.glob(make_path(OUTPUT_PATH, subfolder, elt)):
                os.rename(filename, TEST_PATH + filename.split(OUTPUT_PATH)[-1])


def execute():
    # This function will generate the pictures/marker
    # /!\ All of files inside the both subfolders of the Output folder will be delete before.

    dataWithLabel_Generator()
    extract_test(test_split())

    print("\nAugmentation done.\n")
