# AE_hybrid_encoder.py

import math
import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance

from modules_hybrid_encoder import WeConv1D, IWeConv1D
from bitEstimator_hybrid import BitEstimator

class HybridEncoder(nn.Module):
   
    def __init__(self, d, k):
        super().__init__()
        self.local_feature_extractor = nn.Sequential(
            WeConv1D(3, 64, stride=2),
            WeConv1D(64, 128, stride=2),
            WeConv1D(128, 256, stride=2)
        )
        self.global_aggregator = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, d, 1)
        )
    def forward(self, x_patches):
        local_features = self.local_feature_extractor(x_patches)
        pooled_features = torch.max(local_features, 2, keepdim=True)[0]
        global_feature_vector = self.global_aggregator(pooled_features)
        return global_feature_vector

class SymmetricDecoder(nn.Module):
    
    def __init__(self, d, k):
        super().__init__()
        # K_after_encoding = k / (2*2*2) = k / 8
        self.k_encoded = k // 8 
        
        self.initial_mlp = nn.Sequential(
            nn.Linear(d, 256 * self.k_encoded),
            nn.ReLU(inplace=True)
        )
        
        self.upsampling_stack = nn.Sequential(
            IWeConv1D(256, 128, stride=2),
            IWeConv1D(128, 64, stride=2),
            IWeConv1D(64, 3, stride=2)
        )

    def forward(self, x):
        x = self.initial_mlp(x)
        x = x.view(x.shape[0], 256, self.k_encoded)
        reconstructed_patches = self.upsampling_stack(x)
        return reconstructed_patches

class get_model(nn.Module):
    def __init__(self, k, d):
        super(get_model, self).__init__()
        self.k = k
        self.d = d
        self.encoder = HybridEncoder(d=d, k=k)
        self.decoder = SymmetricDecoder(d=d, k=k)
        self.bit_estimator = BitEstimator(channel=d)

    def forward(self, xyz):
        B, K, C = xyz.shape
        xyz_transposed = xyz.transpose(1, 2)
        
        feature_vector = self.encoder(xyz_transposed).squeeze(-1)

        if self.training:
            noise = torch.empty_like(feature_vector).uniform_(-0.5, 0.5)
            quantized_feature = feature_vector + noise
        else:
            quantized_feature = torch.round(feature_vector)
        
        reconstructed_patches_transposed = self.decoder(quantized_feature)
        reconstructed_patches = reconstructed_patches_transposed.transpose(1, 2)

        mu, scale = self.bit_estimator(quantized_feature)
        dist = torch.distributions.Normal(mu, scale)
        prob = dist.cdf(quantized_feature + 0.5) - dist.cdf(quantized_feature - 0.5)
        total_bits = torch.sum(torch.clamp(-torch.log(prob + 1e-9) / math.log(2.0), 0, 50))
        bpp = total_bits / (K * B)

        return reconstructed_patches, bpp

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, bpp, lamda):
        dist, _ = chamfer_distance(pred, target)
        loss = dist + lamda * bpp
        return loss