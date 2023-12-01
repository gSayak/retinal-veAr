# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Net.block import TransformerLikeNetwork, OptimizeModule, ConvolutionLikeNetwork, BaseUnet


class OpenAccessNet(nn.Module):
    def __init__(self, MVE, MAE, MFI, inputFeature, outputFeature, alpha):
        super(OpenAccessNet, self).__init__()

        if MVE:
            self.vesselSegmentation = ConvolutionLikeNetwork(2, 16, numFeature=64)
        else:
            self.vesselSegmentation = BaseUnet(2, 16, numFeature=64)

        if MAE:
            self.avClassification = TransformerLikeNetwork(inputFeature, 16, numFeature=64)
        else:
            self.avClassification = BaseUnet(inputFeature, 16, numFeature=64)

        self.vesselFinalConv = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )

        if MFI:
            self.optimize = OptimizeModule(alpha)
            self.avFinalConv = nn.Conv2d(32, outputFeature, kernel_size=3, padding=1)
        else:
            self.optimize = None
            self.avFinalConv = nn.Conv2d(16, outputFeature, kernel_size=3, padding=1)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.MFI = MFI

    def forward(self, x, y):
        vessel_tmp = self.vesselSegmentation(y)
        vessel = self.sigmoid(self.vesselFinalConv(vessel_tmp))

        av_tmp = self.avClassification(torch.cat([x, vessel], dim=1))

        if self.MFI:
            res = av_tmp * self.optimize(vessel) * vessel
            av = self.avFinalConv(torch.cat([res, vessel_tmp], dim=1))
        else:
            av = self.avFinalConv(av_tmp)

        return self.softmax(av), vessel
