# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Net.block import TransformerLikeNetwork, OptimizeModule, ConvolutionLikeNetwork


class OpenAccessNet(nn.Module):
    def __init__(self, inputFeature, outputFeature, alpha):
        super(OpenAccessNet, self).__init__()
        self.vesselSegmentation = ConvolutionLikeNetwork(2, 16, numFeature=64)
        self.avClassification = TransformerLikeNetwork(inputFeature, 16, numFeature=64)
        self.avFinalConv = nn.Conv2d(16, outputFeature, kernel_size=3, padding=1)
        self.vesselFinalConv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.optimize = OptimizeModule(alpha)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        vessel = self.sigmoid(self.vesselFinalConv(self.vesselSegmentation(y)))
        av = self.avFinalConv(self.avClassification(torch.cat([x, vessel], dim=1)) * self.optimize(vessel))

        return self.softmax(av), vessel
