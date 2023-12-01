# -*- coding: utf-8 -*-
import torch.nn as nn


class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.net = nn.Conv2d(40, 40, 3, padding=1)

    def forward(self, x):
        return self.net(x)


class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.net = nn.ModuleList()
        for _ in range(4):
            self.net.append(nn.Conv2d(10, 10, 3, padding=1))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    n1 = net1()
    n2 = net2()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(n1))
    print(count_parameters(n2))
    print(count_parameters(n1) / count_parameters(n2))
