import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, fields_dims, num_factors):
        super(FM, self).__init__()
        self.fields_dims = fields_dims
        self.num_factors = num_factors
        num_inputs = int(sum(self.fields_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs,1)
        self.linear = nn.Linear(1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        square_of_sum = torch.sum(self.embedding(x) , dim=1)**2
        sum_of_square = torch.sum(self.embedding(x)**2, dim=1)
        x = self.linear(self.fc(x).sum(1))+0.5*(square_of_sum-sum_of_square).sum(1, keepdims=True)
        x = self.sigmoid(x)
        return x.squeeze()