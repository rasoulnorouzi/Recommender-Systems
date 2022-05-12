import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, positive, negative):
        distance = positive - negative
        loss = - F.logsigmoid(distance).sum()
        return loss