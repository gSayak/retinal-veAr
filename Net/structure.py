# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class ConvolutionPart(nn.Module):
    def __init__(self, channels, kernels=3, strides=1, layers_size=1, isRelu=True):
        """
        :param channels: Including the input channel, marked with channels[0].
        :param kernels: Items used to compute the parameter "padding".
        :param strides: Items used to compute the parameter "padding".
        :param layers_size: Numbers of layers of module.

        Warning: Make sure that "len(channels) == layers_size + 1" and "len(kernels) == layers_size".
        """
        super().__init__()

        if isinstance(kernels, int):
            kernels = [kernels] * layers_size
        if isinstance(strides, int):
            strides = [strides] * layers_size

        assert type(channels) == list, \
            'Something wrong with the type of the parameter "channels" or the length of it. Please check it.'
        assert type(kernels) == list and len(kernels) == layers_size, \
            'Something wrong with the type of the parameter "kernels" or the length of it. Please check it.'
        assert type(strides) == list and len(strides) == layers_size, \
            'Something wrong with the type of the parameter "strides" or the length of it. Please check it.'

        layers = []

        for i in range(layers_size):
            if i >= len(channels) - 1:
                layers.append(nn.Conv2d(channels[-1], channels[-1], kernel_size=kernels[i],
                                        stride=strides[i], padding=(kernels[i] - 1) // 2))
                layers.append(nn.BatchNorm2d(channels[-1]))
            else:
                layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[i],
                                        stride=strides[i], padding=(kernels[i] - 1) // 2))
                layers.append(nn.BatchNorm2d(channels[i + 1]))

            if isRelu:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)


class Cot(nn.Module):
    def __init__(self, featureNum, kernel_size=5):
        super(Cot, self).__init__()
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(featureNum, featureNum, kernel_size=kernel_size, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(featureNum),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(featureNum, featureNum, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(featureNum)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * featureNum, 2 * featureNum // factor, 1, bias=False),
            nn.BatchNorm2d(2 * featureNum // factor),
            nn.ReLU(),
            nn.Conv2d(2 * featureNum // factor, kernel_size * kernel_size * featureNum, 1, stride=1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)
        v = self.value_embed(x).view(bs, c, -1)
        y = torch.cat([k1, x], dim=1)
        att = self.attention_embed(y)
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


class MultiFeatureDetectionBlock(nn.Module):
    def __init__(self, numFeature, numSplit=4):
        super(MultiFeatureDetectionBlock, self).__init__()
        self.Conv1 = nn.Conv2d(numFeature, numFeature, kernel_size=1)
        self.numSplit = numSplit
        self.Conv_group = nn.ModuleList()
        for _ in range(numSplit):
            self.Conv_group.append(nn.Conv2d(numFeature // numSplit, numFeature // numSplit, kernel_size=3, padding=1))
        self.Conv2 = nn.Conv2d(numFeature, numFeature, kernel_size=1)
        self.bn = nn.BatchNorm2d(numFeature)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.Conv1(x)
        splitFeatures = torch.chunk(x, self.numSplit, dim=1)
        out = []
        for i, (feature, layer) in enumerate(zip(splitFeatures, self.Conv_group)):
            if i == 0:
                out.append(layer(feature))
            else:
                out.append(layer(feature + out[-1]))
        x = self.Conv2(torch.cat(out, dim=1))

        return self.relu(self.bn(x + residual))
