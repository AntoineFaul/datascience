import numpy as np


def create_binary(a, n = 1):
    binary = []

    for i in range(len(a)):
    	# a : Input array
    	# n : We want n-max element position to be set to 1
        
        out = np.zeros_like(a[i])
        out[np.arange(len(a[i])), np.argpartition(a[i], -n, axis = 1)[:, -n]] = 1
        binary.append(out)

    return binary

def table(lab_pred, mask_test):
    fp , tp, tn, fn, sum_bg, sum_po, sum_wa, sum_di = (0, 0, 0, 0, 0, 0, 0, 0)
    cont_store = []
    classes = 4

    for j in range(len(mask_test)):
        for i in range(classes):
            output_binary = np.array(create_binary(lab_pred[j]))*2
            mask_output_comparison = np.subtract(mask_test[j][:, :, i],output_binary[:, :, i])
            fp = (mask_output_comparison==-2).sum()
            tp = (mask_output_comparison==-1).sum()
            tn = (mask_output_comparison==0).sum()
            fn = (mask_output_comparison==1).sum()
            cont_store.append((fp, tp, tn, fn))
    
    background, polype, wall, dirt = ([], [], [], [])

    for j in range(classes):
        for i in range(len(mask_test)):
            sum_bg += cont_store[i*4][j]
            sum_po += cont_store[i*4+1][j]
            sum_wa += cont_store[i*4+2][j]
            sum_di += cont_store[i*4+3][j]

        background.append(sum_bg)
        polype.append(sum_po)
        wall.append(sum_wa)
        dirt.append(sum_di)

        sum_bg, sum_po, sum_wa, sum_di = (0, 0, 0, 0)

    background = background / sum(background)
    polype = polype / sum(polype)
    wall = wall / sum(wall)
    dirt = dirt / sum(dirt)

    print("\nBackground: " + "FP: " + str(round(background[0], 3)) + " TP: "+ str(round(background[1], 3)) + " TN: " + str(round(background[2], 3)) + " FN: " + str(round(background[3], 3)))
    print("Polype: "+"FP: " + str(round(polype[0], 3)) + " TP: "+ str(round(polype[1], 3)) + " TN: " + str(round(polype[2], 3))+" FN: " + str(round(polype[3], 3)))
    print("Wall: "+"FP: " + str(round(wall[0], 3)) + " TP: " + str(round(wall[1], 3)) + " TN: " + str(round(wall[2], 3)) +" FN: " + str(round(wall[3], 3)))
    print("Dirt: "+"FP: " + str(round(dirt[0], 3)) + " TP: " + str(round(dirt[1], 3)) + " TN: " + str(round(dirt[2], 3)) +" FN: " + str(round(dirt[3], 3)) + '\n')

def overall_table(lab_pred, mask_test):
    fp, tp, tn, fn = (0, 0, 0, 0)

    for j in range(len(mask_test)):
        output_binary = np.array(create_binary(lab_pred[j]))*2
        mask_output_comparison = np.subtract(mask_test[j], output_binary)
        fp += (mask_output_comparison==-2).sum()
        tp += (mask_output_comparison==-1).sum()
        tn += (mask_output_comparison==0).sum()
        fn += (mask_output_comparison==1).sum()

    sum_classses = (fp + tp + tn + fn)

    fp = fp / sum_classses
    tp = tp / sum_classses
    tn = tn / sum_classses
    fn = fn / sum_classses

    print("\nFP: " + str(round(fp, 3)) + " TP: " + str(round(tp, 3)) + " TN: " + str(round(tn, 3)) + " FN: " + str(round(fn, 3)) + '\n')