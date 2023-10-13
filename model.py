import torch
import torch.nn as nn

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.kernel_size = 9
        
        channels = 32
        self.encoder = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=self.kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )        
        self.flatten = nn.Flatten(1)
        
        flat_size = self._infer_flat_size()
        
        self.enc_mu = nn.Sequential(nn.Linear(flat_size, 256),nn.Linear(256,100))
        self.enc_logvar = nn.Sequential(nn.Linear(flat_size, 256),nn.Linear(256,100))
        
        self.decoder = nn.Sequential(
            nn.Linear(100,256),
            nn.Linear(256,flat_size),
            nn.Unflatten(1,(channels,int(flat_size/channels))),
            nn.ConvTranspose1d(channels, channels, kernel_size=self.kernel_size, stride=4,padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, channels, kernel_size=self.kernel_size, stride=4,output_padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, channels, kernel_size=self.kernel_size, stride=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, channels, kernel_size=self.kernel_size, stride=4,output_padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, channels, kernel_size=self.kernel_size, stride=4,output_padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),   
            nn.ConvTranspose1d(channels, channels, kernel_size=self.kernel_size, stride=4, padding=0,output_padding=3),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, 2, kernel_size=self.kernel_size, stride=1, padding=0,output_padding=0)
        )
        # Decoder
        #self.fc3 = nn.Linear(latent_size, hidden_size)
        #self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        encoder_output = self.encoder(x)
        encoder_output = self.flatten(encoder_output)
        return self.enc_mu(encoder_output), self.enc_logvar(encoder_output)
    
    def reparameterize(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z

    def decode(self, z):
        decoder_output = self.decoder(z)
        return decoder_output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def _infer_flat_size(self):
        flat_size = self.encoder(torch.ones(1, *self.input_size))
        flat_size = self.flatten(flat_size)
        
        return flat_size.shape[1]

# Reconstruction + KL divergence losses summed over all elements and batch
class loss_function(nn.Module):
    def __init__(self):
        super(loss_function,self).__init__()

    def forward(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x,x)
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        #KLD = - 0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
        #print(KLD.shape)

        KLD = - 0.5 * torch.sum(1 + logvar - (torch.pow(mu, 2) + torch.exp(logvar)))
        KLD = KLD*1e-3
        
        return MSE + KLD, MSE.detach().to('cpu'), KLD.detach().to('cpu')

# Initialize the VAE
