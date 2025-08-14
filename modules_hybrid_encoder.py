# modules_hybrid_encoder.py

import torch.nn as nn
from pytorch_wavelets import DWT1D, IDWT1D

class WeConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2, wavelet='haar'):
        super().__init__()
        self.dwt = DWT1D(wave=wavelet, J=1, mode='symmetric')
        self.idwt = IDWT1D(wave=wavelet, mode='symmetric')

        self.main_conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.lf_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.hf_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.main_conv(x)
        yl, yh_list = self.dwt(x)
        yh = yh_list[0]
        yl = self.lf_conv(yl)
        yh = self.hf_conv(yh)
        recon = self.idwt((yl, [yh]))
        return self.activation(recon + residual)

class IWeConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2, wavelet='haar'):
        super().__init__()
        self.dwt = DWT1D(wave=wavelet, J=1, mode='symmetric')
        self.idwt = IDWT1D(wave=wavelet, mode='symmetric')

        self.main_tconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)
        
        self.lf_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.hf_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.shortcut = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.main_tconv(x)
        yl, yh_list = self.dwt(x)
        yh = yh_list[0]
        yl = self.lf_conv(yl)
        yh = self.hf_conv(yh)
        recon = self.idwt((yl, [yh]))
        return self.activation(recon + residual)