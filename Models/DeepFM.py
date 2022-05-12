import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()

        self.field_dims = field_dims
        self.num_factors = num_factors
        self.mlp_dims = mlp_dims
        self.drop_rate = drop_rate
        num_inputs = int(sum(field_dims))
        self.embeddings = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Linear(1,1, bias=True)
        self.sigmod = nn.Sigmoid()
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.layers = []

        for dim in mlp_dims:
            self.layers.append(nn.Linear(input_dim, dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(drop_rate))
            input_dim = dim
        self.layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*self.layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        embed_x = self.embeddings(x)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2
        sum_of_square = torch.sum(embed_x ** 2, dim=1)
        inputs = embed_x.view(-1, self.embed_output_dim)
        x = self.linear_layer(self.fc(x).sum(dim=1))+0.5*(square_of_sum-sum_of_square).sum(dim=1, keepdim=True)
        +self.mlp(inputs)
        x =  self.sigmod(x)
        return x.squeeze()