# WavePatch: Wavelet-Driven Patch-Based Compression for Point Clouds

### Overview

This project implements a deep learning-based approach to point cloud geometry compression, drawing inspiration from the methodology of the paper *"Patch-Based Deep Autoencoder for Point Cloud Geometry Compression"*. Our work adapts this concept by using a **hybrid autoencoder architecture** combined with **wavelet transforms** applied to the latent space for enhanced compression performance. This method aims for superior rate-distortion performance, especially at low bitrates, and is designed to guarantee that the reconstructed point cloud has the same number of points as the original input.

---

### Environment

The project is developed using **Python 3.9** and **PyTorch 1.9.0**.

**Other dependencies:**
- `h5py`: For handling HDF5 data files  
- `scikit-learn`: For data processing utilities  
- `PyWavelets`: For performing wavelet transforms  
- `Pytorch3D`: For KNN and Chamfer Loss (if needed)  

---

### Data Preparation

#### ModelNet40

1. Download from: [http://modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu)  
2. Convert .off files to .ply point clouds:

```bash
python ./sample_data.py ./data/ModelNet40 ./data/ModelNet40_pc_8192 --n_point 8192
```

#### ShapeNet

1. Download from: [ShapeNet Official Link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  
2. Convert .off files to .ply point clouds:

```bash
python ./sample_data.py ./data/shapenetcore_partanno_segmentation_benchmark_v0_normal        ./data/ShapeNet_pc_2048 --n_point 2048
```

---

### Training

To train the hybrid autoencoder:

```bash
python ./train_hybrid.py './data/ModelNet40_pc_8192/**/train/*.ply'        './model/trained_model' --N 8192 --d 16
```

---

### Compression

To compress point clouds using a trained model:

```bash
python ./compress_symmetric_hybrid.py './model/trained_model'        './data/ModelNet40_pc_8192/**/test/*.ply'        './data/ModelNet40_pc_8192_compressed'
```

---

### Decompression

To decompress the previously compressed files:

```bash
python ./decompress_symmetric_hybrid.py './model/trained_model'        './data/ModelNet40_pc_8192_compressed'        './data/ModelNet40_pc_8192_decompressed'
```

---

### Evaluation

To evaluate compression quality (Chamfer Distance + BPP):

```bash
python ./eval.py ./path/to/pc_error        './data/ModelNet40_pc_8192/**/test/*.ply'        './data/ModelNet40_pc_8192_compressed'        './data/ModelNet40_pc_8192_decompressed'        './eval/results.csv'
```

---

### Citation

This project is an implementation and extension of the core ideas presented in the following paper:

```bibtex
@incollection{you2021patch,
  title={Patch-Based Deep Autoencoder for Point Cloud Geometry Compression},
  author={You, Kang and Gao, Pan},
  booktitle={ACM Multimedia Asia},
  pages={1--7},
  year={2021}
}
```

We also drew inspiration from the idea of wavelet-domain convolutional architectures implemented in:  
[https://github.com/fengyurenpingsheng/WeConvene](https://github.com/fengyurenpingsheng/WeConvene)
