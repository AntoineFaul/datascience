import os
import cv2
import numpy as np
from random  import choice
from polyps.file_manager import make_path, clean_subfolders, remove_except_files
import matplotlib.pyplot as plt
from config import config

INPUT_PATH = make_path('polyps', 'origin')
OUTPUT_PATHS = [make_path('polyps', 'training'), make_path('polyps', 'validation'), make_path('polyps', 'test')]

SUBFOLDERS = os.listdir(INPUT_PATH)


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
    input_list = os.listdir(make_path(INPUT_PATH, SUBFOLDERS[0]))
    remove_except_files(input_list)

    (train_val_list, test_list) = split(input_list, config['test_split'])
    (training_list, validation_list) = split(train_val_list, config['validation_split'])
    
    return [training_list, validation_list, test_list]

def execute():
    print("\nData Augmentation:\n")

    for output_path in OUTPUT_PATHS:
        print("Clean folder: " + output_path)
        clean_subfolders(output_path)

    lists = split_data()

    print("\nData set % for training:\t" + str(round((1 - config['test_split'] - (config['validation_split']*(1 - config['test_split'])))*100, 2)) + " %")
    print("Data set % for validation:\t" + str(round((config['validation_split']*(1 - config['test_split']))*100, 2)) + " %")
    print("Data set % for testing:   \t" + str(config['test_split']*100) + " %")

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
