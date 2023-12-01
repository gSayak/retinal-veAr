# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Net.block import BaseUnet, TransformerLikeNetwork, OptimizeModule


class OpenAccessNet(nn.Module):
    def __init__(self, inputFeature, outputFeature, alpha):
        super(OpenAccessNet, self).__init__()
        self.vesselSegmentation = BaseUnet(2, 1, numFeature=64, activation='sigmoid')
        self.avClassification = TransformerLikeNetwork(inputFeature, outputFeature, numFeature=64)
        self.optimize = OptimizeModule(alpha)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        vessel = self.vesselSegmentation(y)
        av = self.avClassification(torch.cat([x, vessel], dim=1))

        return self.softmax(av * self.optimize(vessel)), vessel
