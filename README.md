
# WavePatch: Wavelet-Driven Patch-Based Point Cloud Compression

This repository contains code for compressing and decompressing 3D point cloud data using hybrid autoencoders that integrate classic Wavelet Transforms with deep learning methods. The models are evaluated on the ModelNet40 dataset using both rate (Bits per Point) and distortion (Chamfer Distance) metrics.

## 📂 Project Structure

This repo contains two major families of models:

- **Hybrid Encoder (Feature-level wavelet integration)**: Uses Wavelet Convolution (WeConv) layers in the encoder.
- **Latent Wavelet Autoencoder (Latent-domain wavelet transform)**: Applies DWT to the latent vector.

The following scripts are provided:

```
AE_hybrid_encoder.py       # Feature-level wavelet encoder model
AE_latent_wavelet.py       # Latent-domain wavelet model
bitEstimator.py            # Entropy model for single distribution
bitEstimator_hybrid.py     # Entropy model used in hybrid model

train_hybrid_encoder.py    # Train WeConv-based hybrid model
train_latent_wavelet.py    # Train latent wavelet autoencoder

compress_symmetric_hybrid.py     # Compress using hybrid model
decompress_symmetric_hybrid.py   # Decompress hybrid model outputs

compress_latent_wavelet.py       # Compress latent wavelet model
decompress_latent_wavelet.py     # Decompress latent wavelet outputs

eval_chamfer_hybrid.py     # Evaluate hybrid outputs (BPP + CD)
eval_chamfer_wavelet.py    # Evaluate latent wavelet outputs
modules_hybrid_encoder.py  # Contains WeConv and IWeConv definitions
```

## 🚀 Quick Start

### Training

To train the hybrid encoder model:
```bash
python train_hybrid_encoder.py './data/ModelNet40_pc_8192/**/train/*.ply' './model/trained_hybrid' --N 8192 --d 16
```

To train the latent wavelet model:
```bash
python train_latent_wavelet.py './data/ModelNet40_pc_8192/**/train/*.ply' './model/trained_latent' --N 8192 --d 16
```

### Compression

Compress test set using the hybrid model:
```bash
python compress_symmetric_hybrid.py './model/trained_hybrid' './data/ModelNet40_pc_8192/**/test/*.ply' './data/ModelNet40_pc_8192_compressed'
```

Compress using the latent wavelet model:
```bash
python compress_latent_wavelet.py './model/trained_latent' './data/ModelNet40_pc_8192/**/test/*.ply' './data/ModelNet40_pc_8192_compressed'
```

### Decompression

Decompress files using the hybrid model:
```bash
python decompress_symmetric_hybrid.py './model/trained_hybrid' './data/ModelNet40_pc_8192_compressed' './data/ModelNet40_pc_8192_decompressed'
```

Decompress using the latent wavelet model:
```bash
python decompress_latent_wavelet.py './model/trained_latent' './data/ModelNet40_pc_8192_compressed' './data/ModelNet40_pc_8192_decompressed'
```

### Evaluation

Evaluate hybrid model outputs:
```bash
python eval_chamfer_hybrid.py ./path/to/pc_error './data/ModelNet40_pc_8192/**/test/*.ply' './data/ModelNet40_pc_8192_compressed' './data/ModelNet40_pc_8192_decompressed' './eval/results_hybrid.csv'
```

Evaluate latent wavelet model outputs:
```bash
python eval_chamfer_wavelet.py ./path/to/pc_error './data/ModelNet40_pc_8192/**/test/*.ply' './data/ModelNet40_pc_8192_compressed' './data/ModelNet40_pc_8192_decompressed' './eval/results_wavelet.csv'
```

## 📊 Dataset

We use the [ModelNet40](https://modelnet.cs.princeton.edu/) dataset. Point clouds are sampled at 8192 points.

## 🧠 Reference

This project is based on the idea of combining wavelet transforms with deep autoencoders in two ways:

1. **Feature-Level Integration** (WeConv layers)
2. **Latent-Domain Integration** (DWT on latent space)

For more details, see the final report `ENSC861_Report.pdf`.

## 📄 Citation

If you use this work, consider citing the following sources:
- PointNet [Qi et al., 2017]
- Patch-Based Autoencoder [You et al., 2021]
- Wavelet CNN Compression [Fu et al., 2024]

--

Created by Reyhaneh Ahani, 2025 (Simon Fraser University)
