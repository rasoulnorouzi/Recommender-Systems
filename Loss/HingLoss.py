import torch.nn as nn
import torch.nn.functional as F


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, positive, negative, margin=1):
        distance = positive - negative
        loss = torch.sum(torch.max(-distance+margin,0))
        return loss