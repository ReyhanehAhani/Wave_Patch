import torch
import torch.nn as nn
import torch.nn.functional as F

class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.randn(1, channel) * 0.01)
        self.b = nn.Parameter(torch.randn(1, channel) * 0.01)
        if not final:
            self.a = nn.Parameter(torch.randn(1, channel) * 0.01)
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator(nn.Module):
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, final=True)
        self.channel = channel

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

    def get_pmf(self, device='cpu', L=255):
        with torch.no_grad():
            z = torch.arange(-L // 2, L // 2 + 1, device=device).float()
            z = z.unsqueeze(1).repeat(1, self.channel)
            pmf = self.forward(z + 0.5) - self.forward(z - 0.5)
            pmf = pmf.transpose(0, 1)
            pmf = pmf / (pmf.sum(dim=1, keepdim=True) + 1e-9)
            return pmf
