import numpy as np
import math

import torch

class Criterion:
    def __init__(self, device):
        self.device = device
        self.number = 0

    def __call__(self, outputs, labels):
        mu = outputs[:, :3]
        L = self.getTriangularMatrix(outputs)
        L = L.to(self.device)
        dist = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        loss = -dist.log_prob(labels)
        loss = loss.mean()
        return loss

    def getTriangularMatrix(self, outputs):
        elements = outputs[:, 3:9]
        L = torch.zeros(outputs.size(0), elements.size(1)//2, elements.size(1)//2)
        L[:, 0, 0] = torch.exp(elements[:, 0])
        L[:, 1, 0] = elements[:, 1]
        L[:, 1, 1] = torch.exp(elements[:, 2])
        L[:, 2, 0] = elements[:, 3]
        L[:, 2, 1] = elements[:, 4]
        L[:, 2, 2] = torch.exp(elements[:, 5])
        return L

    def getCovMatrix(self, outputs):
        L = self.getTriangularMatrix(outputs)
        Ltrans = torch.transpose(L, 1, 2)
        LL = torch.bmm(L, Ltrans)
        return LL