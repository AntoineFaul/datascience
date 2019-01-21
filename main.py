from neural_networks.segmentation import train as train_segmentation
from neural_networks.pixel_classification import train as train_pixel_classification
from neural_networks.image_classification import train as train_image_classification

RUN_DATA_AUGMENTATION = True


if __name__ == "__main__":
    cont = True
    c = 0

    while (cont):
        print("\nPlease choose a neural network to execute:\n\n1 - Segmentation\n2 - Pixel Classification\n3 - Image Classification\n")

        try:
            c = int(input('Choice = '))

            assert c > 0
            assert c < 4

            cont = False
            print()

        except:
            print("\nERROR: Invalid Input.\n")

    if c == 1:
        train_segmentation.execute(RUN_DATA_AUGMENTATION)
    elif c == 2:
        train_pixel_classification.execute(RUN_DATA_AUGMENTATION)
    else:

        cont = True
        c = 0

        while (cont):
            print("\nPlease choose a model to execute:\n\n1 - learnOpenCV_model\n2 - reduced_model\n3 - final_model\n4 - reduced_dropOut_model\n5 - reduced_generator_model")

            try:
                c = int(input('Choice = '))

                assert c > 0
                assert c < 6

                cont = False
                print()

            except:
                print("\nERROR: Invalid Input.\n")

        if c == 1:
            train_image_classification.execute("CV")
        elif c == 2:
            train_image_classification.execute("RM")
        elif c == 3:
            train_image_classification.execute("FM")
        elif c == 4:
            train_image_classification.execute("RDM")
        elif c == 5:
            train_image_classification.execute("RGM")
