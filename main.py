# -*- coding: utf-8 -*-

import argparse
import os

from test import test
from train import train
from utils import get_config, get_model, get_test_model
from tools.process import Process

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--mode', default='train')
    arg.add_argument('--preprocess', default=False)
    arg.add_argument('--jsonName', default='./config.json')
    arg = arg.parse_args()

    config, preConfig = get_config(arg.jsonName, arg.mode)
    p = Process(preConfig)
    net, modelName = get_model(arg.jsonName)
    if arg.mode == 'train':
        if arg.preprocess:
            p.run_train()
        train(config, net, modelName)
    else:
        p.run_test()
        testModelNames = [
            'MVE_MAE_MFI', 'MVE_U_MFI', 'MVE_U_None', 'MVE_MAE_None', 'U_U_None', 'U_MAE_MFI', 'U_MAE_None'
        ]

        for modelName in testModelNames:
            if modelName not in os.listdir('./model'):
                continue
            else:
                net = get_test_model(modelName)
                test(config, net, modelName)
