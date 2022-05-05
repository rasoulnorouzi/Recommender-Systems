#%%
from cgi import test
from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from zmq import device



class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, epsilon_std=1.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.h = nn.Linear(input_dim, hidden_dim)
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.h_decoder = nn.Linear(latent_dim, hidden_dim)
        self.x_bar = nn.Linear(hidden_dim, input_dim)


    def forward(self, x):
        h = self.tanh(self.h(x))
        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z = self.sampling(z_mean, z_log_var)
        h_decoder = self.tanh(self.h_decoder(z))
        x_bar = self.sigmoid(self.x_bar(h_decoder))
        return x_bar, z_mean, z_log_var

    def sampling(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.size()).to(device)

        return z_mean + epsilon * torch.exp(z_log_var / 2) * self.epsilon_std


# VAE Loss CE+KL
class VAE_Loss(nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()

    def forward(self, x, x_recon, z_mean, z_log_var):
        BCE = F.binary_cross_entropy(x_recon, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return BCE + KLD


