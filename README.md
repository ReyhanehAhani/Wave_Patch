# WavePatch: Wavelet-Driven Patch-Based Compression for Point Clouds

### Overview

This project investigates two deep learning approaches for efficient point cloud compression, both incorporating wavelet transforms:

- The first approach, a **hybrid encoder**, integrates **Wavelet Convolution (WeConv)** layers to capture multi-scale local features, combined with a **PointNet-inspired module** for global shape context.
- The second approach applies a **wavelet transform directly to the encoder’s latent representation**, followed by quantization and inverse transformation. This strategy exploits wavelets for **frequency-domain decorrelation**, resulting in **sparser, more compressible features**.

By comparing these methods against a baseline patch-based autoencoder, we evaluate rate-distortion performance and visual quality, demonstrating that **latent-space wavelet integration offers superior compression efficiency without sacrificing reconstruction fidelity**.

---

### Environment

- Python 3.9
- PyTorch 1.9.0

**Dependencies:**
- `h5py`
- `scikit-learn`
- `PyWavelets`
- `PyTorch3D`

---

### Project Scripts

```
AE_hybrid_encoder.py            # Feature-level wavelet encoder model (WeConv)
AE_latent_wavelet.py            # Latent-domain wavelet model (DWT on latent code)
bitEstimator.py                 # Entropy model for single distribution
bitEstimator_hybrid.py          # Entropy model used in hybrid model

train_hybrid_encoder.py         # Train WeConv-based hybrid model
train_latent_wavelet.py         # Train latent wavelet autoencoder

compress_symmetric_hybrid.py    # Compress using hybrid model
decompress_symmetric_hybrid.py  # Decompress hybrid model outputs

compress_latent_wavelet.py      # Compress latent wavelet model
decompress_latent_wavelet.py    # Decompress latent wavelet outputs

eval_chamfer_hybrid.py          # Evaluate hybrid outputs (BPP + CD)
eval_chamfer_wavelet.py         # Evaluate latent wavelet outputs
modules_hybrid_encoder.py       # Contains WeConv and IWeConv definitions
```

---

### Training

**Train the hybrid encoder:**
```bash
python train_hybrid_encoder.py './data/ModelNet40_pc_8192/**/train/*.ply' ./model/trained_hybrid --N 8192 --d 16
```

**Train the latent wavelet model:**
```bash
python train_latent_wavelet.py './data/ModelNet40_pc_8192/**/train/*.ply' ./model/trained_latent --N 8192 --d 16
```

---

### Compression

**Hybrid model compression:**
```bash
python compress_symmetric_hybrid.py ./model/trained_hybrid     './data/ModelNet40_pc_8192/**/test/*.ply'     ./data/ModelNet40_pc_8192_compressed
```

**Latent wavelet compression:**
```bash
python compress_latent_wavelet.py ./model/trained_latent     './data/ModelNet40_pc_8192/**/test/*.ply'     ./data/ModelNet40_pc_8192_compressed
```

---

### Decompression

**Hybrid model decompression:**
```bash
python decompress_symmetric_hybrid.py ./model/trained_hybrid     ./data/ModelNet40_pc_8192_compressed     ./data/ModelNet40_pc_8192_decompressed
```

**Latent wavelet decompression:**
```bash
python decompress_latent_wavelet.py ./model/trained_latent     ./data/ModelNet40_pc_8192_compressed     ./data/ModelNet40_pc_8192_decompressed
```

---

### Evaluation

**Evaluate hybrid model outputs:**
```bash
python eval_chamfer_hybrid.py ./path/to/pc_error     './data/ModelNet40_pc_8192/**/test/*.ply'     ./data/ModelNet40_pc_8192_compressed     ./data/ModelNet40_pc_8192_decompressed     ./eval/results_hybrid.csv
```

**Evaluate latent wavelet model outputs:**
```bash
python eval_chamfer_wavelet.py ./path/to/pc_error     './data/ModelNet40_pc_8192/**/test/*.ply'     ./data/ModelNet40_pc_8192_compressed     ./data/ModelNet40_pc_8192_decompressed     ./eval/results_wavelet.csv
```

---

### Citation

```bibtex
@incollection{you2021patch,
  title={Patch-Based Deep Autoencoder for Point Cloud Geometry Compression},
  author={You, Kang and Gao, Pan},
  booktitle={ACM Multimedia Asia},
  pages={1--7},
  year={2021}
}

@article{fu2024weconvene,
  title={WeConvene: Learned Image Compression with Wavelet-Domain Convolution and Entropy Model},
  author={Fu, Hao and Liang, Jie and others},
  year={2024}
}
```

Inspired by [PCC_Patch](https://github.com/I2-Multimedia-Lab/PCC_Patch) and [WeConvene](https://github.com/fengyurenpingsheng/WeConvene)
