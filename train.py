# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import MultiLoss
from utils import count_parameters, adjust_lr, Logger, montage, restore_vessel
from dataload.loader import EyeBallData
from dataload.trans import decomposition

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def train(config, model, modelName, isValidation):

    model = model.to(config.device)

    trainSet = EyeBallData(config, 'train')
    trainLoader = DataLoader(trainSet, shuffle=True, batch_size=config.batchSize)

    valLoader = DataLoader(EyeBallData(config, 'validation'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = MultiLoss(config)

    logger = Logger(f'./log/{modelName}.log')
    os.makedirs(f'./model/{modelName}', exist_ok=True)

    validateSavePath = os.path.join(config.savePath, modelName, 'validation')
    modelSavePath = os.path.join(config.savePath, modelName, 'model')
    if not os.path.exists(modelSavePath):
        os.makedirs(modelSavePath)

    standardAvAcc = 0.8
    standardVesselAcc = 0.9124
    standardAvF1 = 0.8
    standardVesselSen = 0.7

    print('Network scale: ' + str(np.round(count_parameters(model) / 1000000, 3)) + 'M')
    logger.logger.info('Network scale: ' + str(np.round(count_parameters(model) / 1000000, 3)) + 'M')
    print('Start training ......')

    for epoch in range(config.epoch):
        model.train()
        showLoss = 0
        allLoss = 0

        if (epoch + 1) % config.decayEpoch == 0:
            adjust_lr(optimizer, config.decayRate)

        if not os.path.exists(os.path.join(validateSavePath, str(epoch + 1))):
            os.makedirs(os.path.join(validateSavePath, str(epoch + 1), 'av'))
            os.makedirs(os.path.join(validateSavePath, str(epoch + 1), 'ves'))

        with tqdm(total=len(trainLoader), ncols=80, desc=f'[{epoch + 1}/{config.epoch}]: Loss=inf') as bar:
            for i, data in enumerate(trainLoader):
                imgAv, imgVes, lab, ves = data['imgAv'], data['imgVes'], data['labelAv'], data['labelVes']
                if torch.cuda.is_available():
                    imgAv, imgVes = imgAv.to(config.device), imgVes.to(config.device)
                    lab, ves = lab.to(config.device), ves.to(config.device)

                imgAv, imgVes = Variable(imgAv), Variable(imgVes)
                lab, ves = Variable(lab.type(torch.long)), Variable(ves.type(torch.float))

                optimizer.zero_grad()

                predAv, predVessel = model(imgAv, imgVes)

                loss = criterion(predAv, predVessel, lab, ves)

                loss.backward()
                optimizer.step()

                bar.set_description(f'[{epoch + 1}/{config.epoch}]: Loss={np.round(loss.item(), 3)}')

                allLoss += loss.item()
                showLoss += allLoss

                bar.update(1)

        print('loss: %.5f' % showLoss)
        logger.logger.info(f'[{epoch + 1}] total loss: {showLoss}')

        if isValidation:

            model.eval()

            avAcc, avF1, vesselAcc, vesselSen = [], [], [], []

            for i, (data, originalSize, currentSize, name) in enumerate(valLoader):

                imgAvList, imgVesList, lab, ves = data['imgAv'], data['imgVes'], data['labelAv'], data['labelVes']
                outAvList, outVesList = [], []
                lab, ves = lab.numpy().squeeze(), ves.numpy().squeeze()

                originalSize = tuple(map(int, originalSize))
                currentSize = tuple(map(int, currentSize))

                for imgAv, imgVes in zip(imgAvList, imgVesList):
                    imgAv, imgVes = imgAv.to(config.device), imgVes.to(config.device)
                    imgAv, imgVes = Variable(imgAv), Variable(imgVes)

                    predAv, predVessel = model(imgAv, imgVes)

                    predVessel = predVessel[0][0]

                    _, predAv = torch.max(predAv.cpu().data, 1)
                    predAv = predAv[0]

                    predVessel = predVessel.cpu().detach().numpy()

                    outAvList.append(predAv)
                    outVesList.append(predVessel)

                predAv, predVessel = montage(outAvList, outVesList, currentSize)

                savePredAv = Image.fromarray(np.uint8(np.round(cv2.resize(restore_vessel(predAv), originalSize)))).convert('RGB')
                savePredVessel = Image.fromarray(np.uint8(np.round(cv2.resize(predVessel * 255, originalSize)))).convert('L')

                computePredAv, computePredVessel = decomposition(savePredAv), np.round(np.array(savePredVessel) / 255)

                savePredAv.save(os.path.join(validateSavePath, str(epoch + 1), 'av', name[0]))
                savePredVessel.save(os.path.join(validateSavePath, str(epoch + 1), 'ves', name[0]))

                classes = [2, 3]
                TP = ((computePredAv == classes[1]) & (lab == classes[1])).sum()
                TN = ((computePredAv == classes[0]) & (lab == classes[0])).sum()
                FN = ((computePredAv == classes[0]) & (lab == classes[1])).sum()
                FP = ((computePredAv == classes[1]) & (lab == classes[0])).sum()

                if TP + TN + FN + FP == 0:
                    avAcc.append(0)
                    avF1.append(0)
                else:
                    avAcc.append((TP + TN) / (TP + TN + FN + FP))
                    avF1.append((2 * TP) / (2 * TP + FN + FP))

                TP = ((computePredVessel == 1) & (ves == 1)).sum()
                TN = ((computePredVessel == 0) & (ves == 0)).sum()
                FP = ((computePredVessel == 1) & (ves == 0)).sum()
                FN = ((computePredVessel == 0) & (ves == 1)).sum()

                if TP + TN + FN + FP == 0:
                    vesselAcc.append(0)
                    vesselSen.append(0)
                else:
                    vesselAcc.append((TP + TN) / (TP + TN + FN + FP))
                    vesselSen.append(TP / (FN + TP))

            avAcc = np.round(np.array(avAcc).mean(), 4)
            vesselAcc = np.round(np.array(vesselAcc).mean(), 4)

            avF1 = np.round(np.array(avF1).mean(), 4)
            vesselSen = np.round(np.array(vesselSen).mean(), 4)

            if is_better([standardAvAcc, standardAvF1, standardVesselSen, standardVesselAcc],
                         [avAcc, avF1, vesselSen, vesselAcc]):

                path_checkpoint = os.path.join(modelSavePath, 'BestModel_{}_epoch_{}.pth'.format(epoch + 1, avAcc))
                torch.save(model.state_dict(), path_checkpoint)

                print(f'save best model, av val acc: {avAcc}, vessel val acc: {vesselAcc}, '
                      f'av val F1: {avF1}, vessel val sen: {vesselSen}')

                logger.logger.info(
                    f'[{epoch + 1}] best, Av acc: {avAcc}, Vessel acc: {vesselAcc}, Av F1: {avF1}, Vessel sen: {vesselSen}')
                standardAvAcc = avAcc
                standardVesselAcc = vesselAcc
                standardAvF1 = avF1
                standardVesselSen = vesselSen
            else:
                print(f'val acc: {avAcc}, vessel val acc: {vesselAcc}, av val F1: {avF1}, vessel val sen: {vesselSen}')
                logger.logger.info(
                    f'[{epoch + 1}] Av acc: {avAcc}, Vessel acc: {vesselAcc}, Av F1: {avF1}, Vessel sen: {vesselSen}')
        else:
            if epoch + 1 >= 30:
                path_checkpoint = os.path.join(modelSavePath, 'epoch_{}_checkpoint.pth'.format(epoch + 1))
                torch.save(model.state_dict(), path_checkpoint)

    print('Training Completed')


def is_better(standardMetrics, metrics):
    sign = 0
    for i, j in zip(standardMetrics, metrics):
        if i < j:
            sign += 1
        if sign == len(standardMetrics) // 2:
            return True
    return False
