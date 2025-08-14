import torch
import torch.nn as nn

class BitEstimator(nn.Module):
   
    
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel, channel * 2) 
        )

    def forward(self, x):
        
        params = self.net(x)
        mu, scale = params.chunk(2, dim=1)
        mu = torch.clamp(mu, min=-10.0, max=10.0)
        scale = torch.clamp(scale, min=1e-4, max=5.0)

        return mu, scale