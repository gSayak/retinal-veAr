# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
import argparse
from logging import handlers

from Net.net import OpenAccessNet


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(th)


def get_config(name, sign):
    config = argparse.ArgumentParser()
    preConfig = argparse.ArgumentParser()

    with open(name, 'r') as f:
        dic = json.load(f)
    if sign == 'train':
        setting = dict(dic['setting'], **dic[sign])
    else:
        setting = dic['setting']

    preprocessSets = dic['preprocess']

    for k in setting.keys():
        config.add_argument('--' + k, default=setting[k])

    for k in preprocessSets.keys():
        preConfig.add_argument('--' + k, default=preprocessSets[k])

    config = config.parse_args()
    preConfig = preConfig.parse_args()

    return config, preConfig


def get_model(name):
    with open(name, 'r') as f:
        dic = json.load(f)
    para = dic['net']
    datasets = dic['preprocess']['dataSets']
    datasetUsage = '_'.join(datasets) if type(datasets) == list else datasets
    moduleUsage = [datasetUsage, 'MVE' if para['MVE'] else 'U', 'MAE' if para['MAE'] else 'U', 'MFI' if para['MFI'] else 'None']

    return OpenAccessNet(**para), '_'.join(moduleUsage)


def get_test_model(name):
    moduleUsage = name.split('_')

    para = {"MVE": True if 'MVE' in moduleUsage else False,
            "MAE": True if 'MAE' in moduleUsage else False,
            "MFI": True if 'MFI' in moduleUsage else False,

            "inputFeature": 4,
            "outputFeature": 4,

            "alpha": 1.0}

    return OpenAccessNet(**para)


def restore_vessel(data):
    r = np.zeros_like(data)
    g = np.zeros_like(data)
    b = np.zeros_like(data)

    r[data == 3] = 255
    g[data == 1] = 255
    b[data == 2] = 255

    r, g, b = np.expand_dims(r, 2), np.expand_dims(g, 2), np.expand_dims(b, 2)

    return np.concatenate((r, g, b), axis=2)


def adjust_lr(optimize, decay_rate):
    for param_group in optimize.param_groups:
        param_group['lr'] *= decay_rate
    print('The learning rate has changed')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def montage(outAvList, outVesList, size):
    w, h = size
    s = outAvList[0].shape[0]

    m = w // s + (0 if w % s == 0 else 1)
    n = h // s + (0 if h % s == 0 else 1)

    tmp_w, tmp_h = (m * s - w) // 2, (n * s - h) // 2

    predAv = np.zeros((n * s, m * s))
    predVessel = np.zeros((n * s, m * s))

    for j in range(m):
        for i in range(n):
            predAv[i * s: (i + 1) * s, j * s: (j + 1) * s] = outAvList[j * n + i]
            predVessel[i * s: (i + 1) * s, j * s: (j + 1) * s] = outVesList[j * n + i]

    predAv, predVessel = predAv[tmp_h: tmp_h + h, tmp_w: tmp_w + w], predVessel[tmp_h: tmp_h + h, tmp_w: tmp_w + w]

    return predAv, predVessel
