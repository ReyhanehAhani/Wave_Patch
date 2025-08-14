# compress_symmetric_hybrid.py

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
from pytorch3d.ops.knn import knn_points
import torchac


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import AE_hybrid_encoder as AE

import pc_kit
import pc_io

def get_cdf_tables(mu, scale, min_val, max_val):
    symbols = torch.arange(min_val, max_val + 1, device=mu.device, dtype=torch.float32)
    mu_r = mu.unsqueeze(-1)
    scale_r = scale.unsqueeze(-1)
    symbols_r = symbols.view(1, 1, -1)
    dist = torch.distributions.Normal(mu_r, scale_r)
    cdfs = dist.cdf(symbols_r)
    cdfs_with_tails = torch.cat([
        torch.zeros_like(cdfs[..., :1]),
        cdfs,
        torch.ones_like(cdfs[..., :1])
    ], dim=-1)
    return cdfs_with_tails

def main(args):
    if not os.path.exists(args.compressed_path):
        os.makedirs(args.compressed_path)

    files = pc_io.get_files(args.glob_input_path)
    filenames = [os.path.splitext(os.path.basename(x))[0] for x in files]
    print(f"Found {len(files)} point clouds to compress.")

    print("Loading symmetric hybrid model...")
    ae_model = AE.get_model(k=args.K, d=args.d).cuda().eval()
    ae_model.load_state_dict(torch.load(args.model_path))
    print("Model loaded successfully.")

    min_val, max_val = -50, 50

    with torch.no_grad():
        for i, filepath in enumerate(tqdm(files, desc="Compressing")):
            pc = pc_io.load_points([filepath], p_min=0, p_max=1, processbar=False)
            pc = torch.Tensor(pc).cuda()
            N = pc.shape[1]
            S = N * args.ALPHA // args.K

            sampled_xyz = pc_kit.index_points(pc, pc_kit.farthest_point_sample_batch(pc, S))
            _, _, grouped_xyz = knn_points(sampled_xyz, pc, K=args.K, return_nn=True)
            x_patches = grouped_xyz - sampled_xyz.view(1, S, 1, 3)
            batch_of_patches = x_patches.squeeze(0)

            feature_vectors = ae_model.encoder(batch_of_patches.transpose(1, 2)).squeeze(-1)
            quantized_features = torch.round(feature_vectors)
            quantized_features = torch.clamp(quantized_features, min=min_val, max=max_val)

            mu, scale = ae_model.bit_estimator(quantized_features)
            cdf_tables = get_cdf_tables(mu, scale, min_val, max_val)
            
            symbols_to_encode = quantized_features.to(torch.int16)
            symbols_for_coder = (symbols_to_encode - min_val).cpu()
            byte_stream = torchac.encode_float_cdf(cdf_tables.cpu(), symbols_for_coder, check_input_bounds=True)

            with open(os.path.join(args.compressed_path, filenames[i] + '.p.bin'), 'wb') as fout:
                fout.write(byte_stream)

            sampled_xyz_to_save = sampled_xyz.squeeze(0).cpu().numpy().astype(np.float16)
            sampled_xyz_to_save.tofile(os.path.join(args.compressed_path, filenames[i] + '.s.bin'))

            header = np.array([S, args.K, args.d], dtype=np.uint16)
            header.tofile(os.path.join(args.compressed_path, filenames[i] + '.h.bin'))

    print("\nCompression finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='compress_symmetric_hybrid.py',
        description='Compress Point Clouds using the trained Symmetric Hybrid Autoencoder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', help='Path to the trained model state_dict (.pth file).')
    parser.add_argument('glob_input_path', help='Point clouds glob pattern to be compressed.')
    parser.add_argument('compressed_path', help='Directory to save compressed files.')
    parser.add_argument('--K', type=int, help='Number of points in each patch (must match training).', default=1024)
    parser.add_argument('--d', type=int, help='Bottleneck size (must match training).', default=32)
    parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
    
    args = parser.parse_args()
    main(args)