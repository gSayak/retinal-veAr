# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from Net.structure import ConvolutionPart, Cot, MultiFeatureDetectionBlock


class BaseUnet(nn.Module):
    """
    Unet:
        default: 16, 32, 64, 128, 256
    """

    def __init__(self, inputFeature, outputFeature, numFeature=16, activation=None):
        super(BaseUnet, self).__init__()

        n = numFeature

        filters = [inputFeature]
        for i in range(5):
            filters.append((2 ** i) * n)

        self.EncodeBlock = nn.ModuleList()
        self.DecodeBlock = nn.ModuleList()

        for i in range(len(filters) - 2):
            self.EncodeBlock.append(ConvolutionPart([filters[i], filters[i + 1]], layers_size=2))

        for i in range(1, len(filters) - 1):
            self.DecodeBlock.append(nn.Upsample(scale_factor=2))
            self.DecodeBlock.append(ConvolutionPart([filters[-i] + filters[-i - 1], filters[-i - 1]], layers_size=2))

        self.MaxPool = nn.MaxPool2d(2)
        self.middleConv = ConvolutionPart([filters[-2], filters[-1]], layers_size=2)
        self.finalConv = nn.Conv2d(n, outputFeature, kernel_size=3, stride=1, padding=1)

        if activation is not None:
            self.isActivation = True
            if activation == 'softmax':
                self.activation = nn.Softmax(dim=1)
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            else:
                raise
        else:
            self.isActivation = False

    def forward(self, x):
        features = []

        for i in range(len(self.EncodeBlock)):
            x = self.EncodeBlock[i](x)
            features.append(x)
            x = self.MaxPool(x)

        x = self.middleConv(x)

        nextFeatures = []
        for i in range(0, len(self.DecodeBlock), 2):
            x = self.DecodeBlock[i](x)
            x = self.DecodeBlock[i + 1](torch.cat([features[-i // 2 - 1], x], dim=1))
            nextFeatures.append(x)
        nextFeatures.pop()

        res = self.activation(self.finalConv(x)) if self.isActivation else self.finalConv(x)

        return res


class ConvolutionLikeNetwork(nn.Module):

    def __init__(self, inputFeature, outputFeature, numFeature):
        super(ConvolutionLikeNetwork, self).__init__()

        n = numFeature
        filters = [inputFeature]
        for i in range(5):
            filters.append((2 ** i) * n)

        self.EncodeBlock = nn.ModuleList()
        self.DecodeBlock = nn.ModuleList()

        for i in range(len(filters) - 2):
            tmp = nn.Sequential(
                ConvolutionPart([filters[i], filters[i + 1]], layers_size=1),
                MultiFeatureDetectionBlock(filters[i + 1])
            )
            self.EncodeBlock.append(tmp)

        for i in range(1, len(filters) - 1):
            self.DecodeBlock.append(nn.Upsample(scale_factor=2))
            self.DecodeBlock.append(ConvolutionPart([filters[-i] + filters[-i - 1], filters[-i - 1]], layers_size=2))

        self.MaxPool = nn.MaxPool2d(2)

        self.middleConv = ConvolutionPart([filters[-2], filters[-1]])
        self.finalConv = nn.Conv2d(n, outputFeature, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        features = []

        for i in range(len(self.EncodeBlock)):
            x = self.EncodeBlock[i](x)
            features.append(x)
            x = self.MaxPool(x)

        x = self.middleConv(x)

        nextFeatures = []
        for i in range(0, len(self.DecodeBlock), 2):
            x = self.DecodeBlock[i](x)
            x = self.DecodeBlock[i + 1](torch.cat([features[-i // 2 - 1], x], dim=1))
            nextFeatures.append(x)
        nextFeatures.pop()

        return self.finalConv(x)
        

class OptimizeModule(nn.Module):
    def __init__(self, alpha):
        super(OptimizeModule, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        y = self.alpha * (torch.exp(-(torch.abs(x - 0.5))) - torch.exp(torch.tensor(-0.5))) + 1
        return y * x


class TransformerLikeNetwork(nn.Module):
    def __init__(self, inputFeature, outputFeature, numFeature):
        super(TransformerLikeNetwork, self).__init__()
        n = numFeature
        filters = [inputFeature]
        for i in range(5):
            filters.append((2 ** i) * n)

        self.EncodeBlock = nn.ModuleList()
        self.DecodeBlock = nn.ModuleList()

        for i in range(len(filters) - 2):
            tmp = nn.Sequential(
                ConvolutionPart([filters[i], filters[i + 1]], layers_size=1),
                Cot(filters[i + 1])
            )
            self.EncodeBlock.append(tmp)

        for i in range(1, len(filters) - 1):
            tmp = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv2d(filters[-i], filters[-i - 1], kernel_size=1),
                                nn.ReLU())
            self.DecodeBlock.append(tmp)
            self.DecodeBlock.append(ConvolutionPart([filters[-i], filters[-i - 1]], layers_size=2))

        self.MaxPool = nn.MaxPool2d(2)

        self.middleConv = ConvolutionPart([filters[-2], filters[-1]])
        self.finalConv = nn.Conv2d(n, outputFeature, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        features = []

        for i in range(len(self.EncodeBlock)):
            x = self.EncodeBlock[i](x)
            features.append(x)
            x = self.MaxPool(x)

        x = self.middleConv(x)

        nextFeatures = []
        for i in range(0, len(self.DecodeBlock), 2):
            x = self.DecodeBlock[i](x)
            x = self.DecodeBlock[i + 1](torch.cat([features[-i // 2 - 1], x], dim=1))
            nextFeatures.append(x)
        nextFeatures.pop()

        return self.finalConv(x)
