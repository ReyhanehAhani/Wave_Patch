# AE_latent_wavelet.py

import torch
import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D
from pytorch3d.loss import chamfer_distance
from pc_kit import PointNet, SAPP
from bitEstimator import BitEstimator

class LatentWaveletAutoencoder(nn.Module):
    def __init__(self, k, d):
        super(LatentWaveletAutoencoder, self).__init__()
        self.k = k
        self.d = d
        self.sa = SAPP(in_channel=3, feature_region=8, mlp=[32, 64, 128], bn=False)
        self.pn = PointNet(in_channel=3+128, mlp=[256, 512, 1024, d], relu=[True]*3 + [False], bn=False)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, k * 3)
        )
        self.dwt = DWT1D(wave='haar', J=1, mode='symmetric')
        self.idwt = IDWT1D(wave='haar', mode='symmetric')

        with torch.no_grad():
            dummy = torch.randn(1, 1, d)
            yl, yh = self.dwt(dummy)
            self.d_lf = yl.shape[-1]
            self.d_hf = yh[0].shape[-1]

        self.be_lf = BitEstimator(self.d_lf)
        self.be_hf = BitEstimator(self.d_hf)

    def forward(self, xyz):
        B, K_in, _ = xyz.shape
        feature = self.sa(xyz.transpose(1, 2))
        feature = self.pn(torch.cat((xyz.transpose(1, 2), feature), dim=1))
        yl, yh_list = self.dwt(feature.unsqueeze(1))
        yh = yh_list[0]

        if self.training:
            q_yl = yl + torch.empty_like(yl).uniform_(-0.5, 0.5)
            q_yh = yh + torch.empty_like(yh).uniform_(-0.5, 0.5)
        else:
            q_yl = torch.round(yl)
            q_yh = torch.round(yh)

        prob_lf = self.be_lf(q_yl.squeeze(1) + 0.5) - self.be_lf(q_yl.squeeze(1) - 0.5)
        prob_hf = self.be_hf(q_yh.squeeze(1) + 0.5) - self.be_hf(q_yh.squeeze(1) - 0.5)
        bits = torch.sum(torch.clamp(-torch.log2(prob_lf + 1e-10), 0, 50)) + \
               torch.sum(torch.clamp(-torch.log2(prob_hf + 1e-10), 0, 50))
        bpp = bits / (K_in * B)

        latent = self.idwt((q_yl, [q_yh])).squeeze(1)
        if latent.shape[1] != self.d:
            latent = latent[:, :self.d] if latent.shape[1] > self.d else \
                     torch.cat([latent, torch.zeros(B, self.d - latent.shape[1], device=latent.device)], dim=1)

        recon = self.decoder(latent).view(B, self.k, 3)
        return recon, bpp

    def iwt(self, q_yl, q_yh):
        
        return self.idwt((q_yl, [q_yh]))

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    
    def forward(self, pred, target, bpp, lamda):
        dist, _ = chamfer_distance(pred, target)
        return dist + lamda * bpp

def get_model(k, d):
    return LatentWaveletAutoencoder(k=k, d=d)
