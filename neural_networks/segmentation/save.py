#save the images
import matplotlib.pyplot as plt

from data.manager import clean_folder, list_dir, make_path


def write_images(lab_pred, output_path):
    clean_folder(output_path)

    names = list_dir(make_path('polyps', 'test', 'data'))

    for i in range(len(lab_pred)):
        plt.imsave(make_path(output_path, names[i]), lab_pred[i])
