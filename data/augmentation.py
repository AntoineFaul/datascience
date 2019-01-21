import cv2
import numpy as np
from random  import choice

from .manager import make_path, clean_subfolders, list_dir
from config import config

INPUT_PATH = make_path('polyps_pixel', 'origin')
OUTPUT_PATHS = [make_path('polyps_pixel', 'training'), make_path('polyps_pixel', 'validation'), make_path('polyps_pixel', 'test')]

SUBFOLDERS = list_dir(INPUT_PATH)


def rotate(img, rot):
    rows, cols = len(img), len(img[0])

    # Rotation
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), rot, 1)
    dst = cv2.warpAffine(img, M1, (cols, rows))

    # Zoom for black corner removal
    zoom = round((-(1/45)*pow(rot%90 - 45, 2) + 45) / 45 * 0.3, 2)

    pts1 = np.float32([[int(rows*(zoom/2)), int(cols*(zoom/2))], [int(rows-(rows*(zoom/2))), int(cols*(zoom/2))], [int(rows*(zoom/2)), int(cols-(cols*(zoom/2)))], [int(rows-(rows*(zoom/2))), int(cols-(cols*(zoom/2)))]])
    pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])

    M2 = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(dst, M2, (rows, cols))

    dst = cv2.resize(dst, config['image_size'], interpolation = cv2.INTER_CUBIC)

    return dst

def split(input_list, perc):
    split_list = []

    for i in range(int(len(input_list)*perc)):
        im = choice(input_list)
        split_list.append(im)
        input_list.remove(im)

    return (input_list, split_list)

def split_data():
    input_list = list_dir(make_path(INPUT_PATH, SUBFOLDERS[0]))

    (train_val_list, test_list) = split(input_list, config['test_split'])
    (training_list, validation_list) = split(train_val_list, config['validation_split'])
    
    return [training_list, validation_list, test_list]

def print_split_perc():
    split_test_perc = config['test_split']*100
    split_val_perc = round(config['validation_split'] * (100 - split_test_perc), 2)

    print("\nData set % for training:\t" + str(100 - split_val_perc - split_test_perc) + " %")
    print("Data set % for validation:\t" + str(split_val_perc) + " %")
    print("Data set % for testing:   \t" + str(split_test_perc) + " %")


def execute():
    print("\nData Augmentation:\n")

    for output_path in OUTPUT_PATHS:
        print("Clean folder: " + output_path)
        clean_subfolders(output_path)

    print_split_perc()
    lists = split_data()

    for i in range(len(OUTPUT_PATHS)):
        print()
        for j in range(len(lists[i])):
            for k in range(config['multiplier']):
                for subfolder in SUBFOLDERS:
                    print("\r[" + str(i) + "/" + str(len(OUTPUT_PATHS)-1) + "] --> Nb of images: " + str((j+1)*(k+1)), end = '')

                    img = cv2.imread(make_path(INPUT_PATH, subfolder, lists[i][j]))
                    dst = rotate(img, int(k * (360/config['multiplier'])))

                    filename = lists[i][j].split('.')[0]
                    filename += '_' + str(k) + '.jpg'

                    cv2.imwrite(make_path(OUTPUT_PATHS[i], subfolder, filename), dst)

    print("\n\nAugmentation done.\n")
