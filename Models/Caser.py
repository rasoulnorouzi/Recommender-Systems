import torch
import torch.nn as nn



class Caser(nn.Module):
    def __init__(self, num_user, num_items, num_factors, L=5, d=16, d_prime = 4, drop_ratio = 0.05):
        super(Caser, self).__init__()
        '''
        num_user: number of users
        num_items: number of items
        num_factors: number of latent factors
        L: number of layers
        d: number of filters in horizontal convolution
        d_prime: number of filters in vertical convolution
        drop_ratio: dropout ratio

        '''

        # Users Embedding Layer
        self.P = nn.Embedding(num_user, num_factors)

        # Items Sequence Embedding Layer
        self.Q = nn.Embedding(num_items, num_factors)

        # Output chaannels for the V_Conv and H_Conv
        self.d_prime, self.d, self.L = d_prime, d, L

        # Vertical Convolution Layer
        self.conv_v = nn.Conv2d(1, d_prime, (L, 1))

        # Horizontal Convolution Layer
        self.conv_h = nn.ModuleList([ nn.Conv2d( in_channels=1, out_channels = d, kernel_size = (i+1, num_factors)) for i in range(L)])
        # Horizontal MaxPool Layer
        self.max_pool = nn.ModuleList([nn.MaxPool1d(L-i, stride=1) for i in range(L)])

        # Fully Connected Layer
        self.fc = nn.Linear(d_prime*num_factors+d*L, num_factors)

        # Items Embedding Layer
        self.Q_prime = nn.Embedding(num_items, num_factors*2)
        self.b = nn.Embedding(num_items, 1)

        # Activation and Dropout Layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_ratio)

        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d *L

    def forward(self, user_id, seq, item_id):
        item_embs = torch.unsqueeze(self.Q(seq), dim=1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []

        # Vertical Convolution
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        
        # Horizontal Convolution
        if self.d:
            for i in range(self.L):
                conv_out = torch.squeeze(self.relu(self.conv_h[i](item_embs)), dim=3)
                t = self.max_pool[i](conv_out)
                pool_out = torch.squeeze(t, dim=2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, dim=1)

        out = torch.cat([out_v, out_h], dim=1)
        z = self.dropout(self.relu(self.fc(out)))
        x = torch.cat([torch.squeeze(user_emb, dim=1), z], dim=1)

        # Item Embedding
        q_prime_i = torch.squeeze(self.Q_prime(item_id))
        b = torch.squeeze(self.b(item_id))

        # Prediction
        result = (x * q_prime_i).sum(dim=1) + b
        return result