import numpy as np
from PIL import Image

from data.manager import clean_folder, list_dir, make_path
from config import config


def pixel_class(pixel):
    return config['colors']['rgb'][pixel.argmax()]

def merge(array):
    final_images = []

    for image in array:
        cimage = []

        for row in image:
            crow = []

            for pixel in row:
                crow.append(pixel_class(pixel))

            cimage.append(crow)

        final_images.append(cimage)

    return final_images


def write_images(lab_pred, output_path):
    clean_folder(output_path)

    array = merge(lab_pred)
    names = list_dir(make_path('polyps_pixel', 'test', 'data'))
    index = 0

    for image in array:
        img = Image.new('RGB', config['image_size'], 'white')

        for i in range(config['image_max']):
            for j in range(config['image_max']):
                img.putpixel((i, j), image[i][j])

        n = (names[index].split(config['path_sep']))[-1]
        name = '{}'.format(n)

        img.save(make_path(output_path, name))
