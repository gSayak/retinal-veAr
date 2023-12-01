# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class MultiLoss(nn.Module):
    def __init__(self, config, ep=1e-6):
        super(MultiLoss, self).__init__()
        self.ep = ep

        avWeight = torch.FloatTensor([1, 1, 3, 5]).to(config.device)
        self.avCriterion = nn.CrossEntropyLoss(weight=avWeight).to(config.device)

        self.vesselCriterion = nn.BCELoss().to(config.device)

    def dice_loss(self, predVessel, labelVessel):
        intersection = 2 * torch.sum(predVessel * labelVessel) + self.ep
        union = torch.sum(predVessel) + torch.sum(labelVessel) + self.ep
        loss = 1 - intersection / union
        return loss

    def forward(self, predAv, predVessel, labelAv, labelVessel):
        avLoss = vesselLoss = 0

        avLoss += self.avCriterion(predAv, labelAv)

        vesselLoss += self.vesselCriterion(predVessel, labelVessel)
        vesselLoss += 0.1 * self.dice_loss(predVessel, labelVessel)

        return vesselLoss + avLoss
