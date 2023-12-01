# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
from sklearn.metrics import confusion_matrix

from PIL import Image


def composition(image):
    label_av = np.copy(np.asarray(image))
    label = np.zeros_like(label_av[..., 0])

    label[label_av[:, :, 0] == 255] = 3
    label[label_av[:, :, 2] == 255] = 2
    label[label_av[:, :, 1] == 255] = 1

    return label


def compute_for_av(pred, label):

    pred = pred.flatten()
    label = label.flatten()

    ma = confusion_matrix(label, pred)

    TP, TN, FP, FN = ma[3][3], ma[2][2], ma[3][2], ma[2][3]

    return {'avAcc': (TP + TN) / (TP + TN + FN + FP),
            'avSen': TP / (TP + FN),
            'avSpe': TN / (TN + FP),
            'avF1': (2 * TP) / (2 * TP + FP + FN),
            'avPre': TP / (TP + FP),
            'avIou': TP / (TP + FN + FP)
            }


def compute_for_vessel(pred, label):

    pred = np.round(np.array(pred) / 255)
    label = np.array(label) // 255

    pred = pred.flatten()
    label = label.flatten()

    ma = confusion_matrix(label, pred)

    TP, TN, FP, FN = ma[1][1], ma[0][0], ma[1][0], ma[0][1]

    return {'vesAcc': (TP + TN) / (TP + TN + FN + FP),
            'vesSen': TP / (TP + FN),
            'vesSpe': TN / (TN + FP),
            'vesF1': (2 * TP) / (2 * TP + FP + FN),
            'vesPre': TP / (TP + FP),
            'vesIou': TP / (TP + FN + FP)
            }


def compute(predPath, dataset):

    labelPath = f'E:/label/{dataset}_label'

    finalResult = defaultdict(float)
    for imageName in os.listdir(os.path.join(predPath, 'av')):
        predVes = Image.open(os.path.join(predPath, 'ves', imageName)).convert('L')
        predAv = composition(Image.open(os.path.join(predPath, 'av', imageName)).convert('RGB'))

        imageName = imageName.split('_')[-1]

        labelVes = Image.open(os.path.join(labelPath, 'ves', imageName)).convert('L')
        labelAv = composition(Image.open(os.path.join(labelPath, 'av', imageName)).convert('RGB'))

        vesResult = compute_for_vessel(predVes, labelVes)
        avResult = compute_for_av(predAv, labelAv)

        for k, v in vesResult.items():
            finalResult[k] += v

        for k, v in avResult.items():
            finalResult[k] += v

    for k, v in finalResult.items():
        finalResult[k] = v / len(os.listdir(os.path.join(predPath, 'av')))

    return finalResult


if __name__ == '__main__':
    path = r'E:\project\OpenAccess\result\LES_MVE_MAE_MFI'
    result = compute(path, 'LES')
    for k, v in result.items():
        print(k, v)

    # result = OrderedDict()
    # for folder in os.listdir(path):
    #     result[folder] = compute(os.path.join(path, folder), 'DRIVE')
    # df = pd.DataFrame(result)
    # df = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    # df.to_csv('./result.csv')
