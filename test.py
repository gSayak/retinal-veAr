# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataload.loader import EyeBallData
from utils import restore_vessel


def test(config, net, modelName):

    model = net
    parmPath = f'/data2/chenchouyu/OpenAccess/{modelName}/model/'
    bestParm = max([i for i in os.listdir(parmPath) if i.endswith('.pth')], key=lambda x: float(x.split('_')[-1][:-4]))

    print(f'test model: {modelName}, pth name: {bestParm}')

    modelUsage = modelName.split('_')

    model.load_state_dict(torch.load(os.path.join(parmPath, bestParm)), False)
    model = model.to(config.device)
    model.eval()

    testLoad = EyeBallData(config, 'test')
    testLoader = DataLoader(testLoad, shuffle=False)

    os.makedirs(os.path.join(f'./Result/{modelName}', 'av'), exist_ok=True)
    os.makedirs(os.path.join(f'./Result/{modelName}', 'ves'), exist_ok=True)

    with tqdm(total=len(testLoader), desc='Test', ncols=80) as bar:
        for item, (data, originalSize, currentSize, name) in enumerate(testLoader):
            resImgAvDict, resImgVesselDict = data['imgAv'], data['imgVes']

            originalSize = tuple(map(int, originalSize))
            currentSize = tuple(map(int, currentSize))

            res = {}
            for k in resImgAvDict.keys():
                outListAv = []
                outListVes = []
                for idx, (imgAv, imgVes) in enumerate(zip(resImgAvDict[k], resImgVesselDict[k])):
                    imgAv, imgVes = imgAv.to(config.device), imgVes.to(config.device)
                    imgAv, imgVes = Variable(imgAv), Variable(imgVes)

                    predAv, predVessel = model(imgAv, imgVes)

                    if 'MS' in modelUsage:
                        predVessel = predVessel[-1]

                    if 'CA' in modelUsage:
                        predAv = predAv[-1]

                    _, predAv = torch.max(predAv.cpu().data, 1)
                    predAv = predAv[0]

                    predVessel = predVessel[0][0].detach().cpu().numpy()

                    outListAv.append(predAv)
                    outListVes.append(predVessel)

                res[k] = [outListAv, outListVes]

            outAv, outVessel = threshold_vessel(res, currentSize)

            saveAv = Image.fromarray(np.uint8(np.round(cv2.resize(restore_vessel(outAv), originalSize)))).convert('RGB')
            saveVes = Image.fromarray(np.uint8(np.round(cv2.resize(outVessel * 255, originalSize)))).convert('L')

            saveAv.save(os.path.join(f'./Result/{modelName}', 'av', name[0]))
            saveVes.save(os.path.join(f'./Result/{modelName}', 'ves', name[0]))
            bar.update(1)


def threshold_vessel(res, currentSize):
    w, h = currentSize

    s = 256

    m = w // s + 1
    n = h // s + 1

    size = (n * s, m * s)

    outAv, outVessel = [], []
    for k in res.keys():

        newAvImg = np.zeros(size)
        newVesselImg = np.zeros(size)

        for i, out in enumerate(res[k][0]):
            newAvImg[int(i % n) * s: int(i % n) * s + s, int(i / n) * s: int(i / n) * s + s] = out

        for i, out in enumerate(res[k][1]):
            newVesselImg[int(i % n) * s: int(i % n) * s + s, int(i / n) * s: int(i / n) * s + s] = out

        outAv.append(np.round(newAvImg[k[0]: k[0] + h, k[1]: k[1] + w]))

        vesselMax = newVesselImg[k[0]: k[0] + h, k[1]: k[1] + w].max()
        outVessel.append(newVesselImg[k[0]: k[0] + h, k[1]: k[1] + w] / vesselMax)

    av = np.zeros_like(outAv[0])
    for i in range(len(outAv[0])):
        for j in range(len(outAv[0][0])):
            pointSum = [0, 0, 0, 0]
            for tmp in outAv:
                try:
                    pointSum[int(tmp[i][j])] += 1
                except:
                    raise
            av[i][j] = pointSum.index(max(pointSum))

    vessel = np.array(outVessel)
    vessel = vessel.mean(0)

    return av, vessel
