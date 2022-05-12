import torch
import torch.nn as nn



class NeuMF(nn.Module):
    def __init__(self, num_factors, num_users, num_items,num_hiddens):
        super(NeuMF, self).__init__()
        '''
        num_factors: the number of latent factors
        num_users: the number of users
        num_items: the number of items
        num_hiddens: the number of hidden units
        
        '''

        self.weight_init()
        self.sigmoid = nn.Sigmoid()

        self.P = nn.Embedding(num_users+1, num_factors)
        self.Q = nn.Embedding(num_items+1, num_factors)
        self.U = nn.Embedding(num_users+1, num_factors)
        self.V = nn.Embedding(num_items+1, num_factors)
        
        self.layers = [nn.Linear(num_factors*2, num_hiddens[0], bias=True), nn.ReLU()]

        for i in range(len(num_hiddens)-1):
            self.layers.append(nn.Linear(num_hiddens[i], num_hiddens[i+1], bias=True))
            self.layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*self.layers)
        self.prediction_layer = nn.Linear(num_hiddens[-1]+num_factors,1, bias=False)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(torch.cat((p_mlp, q_mlp), dim=1))
        con_res = torch.cat((gmf, mlp), dim=1)
        return self.prediction_layer(con_res)

    def weight_init(self):
         for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.normal_(0, 0.01)
    
    def regularization(self):
        return torch.sum(torch.pow(self.P.weight, 2)) + torch.sum(torch.pow(self.Q.weight, 2))
        + torch.sum(torch.pow(self.U.weight, 2)) + torch.sum(torch.pow(self.V.weight, 2))
