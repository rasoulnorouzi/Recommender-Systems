import torch
import torch.nn as nn




class MF(nn.Module):
    
    def __init__(self, n_users, n_items, n_factors):
        super(MF, self).__init__()
        '''
        n_users: number of unique users
        n_items: number of unique items
        n_factors: number of latent factors
        '''

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_embed = nn.Embedding(n_users+1, n_factors)
        self.item_embed = nn.Embedding(n_items+1, n_factors)
        self.user_bias = nn.Embedding(n_users+1, 1)
        self.item_bias = nn.Embedding(n_items+1, 1)
        self.init_weights()

    def init_weights(self):
        self.user_embed.weight.data.uniform_(-0.01, 0.01)
        self.item_embed.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

    def regularization(self):
        return torch.sum(torch.pow(self.user_embed.weight, 2)) + torch.sum(torch.pow(self.item_embed.weight, 2))

    def forward(self, user, item):
        user_embed = self.user_embed(user)
        item_embed = self.item_embed(item)
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        return torch.sum(user_embed * item_embed, dim=1) + torch.squeeze(user_bias) + torch.squeeze(item_bias)
