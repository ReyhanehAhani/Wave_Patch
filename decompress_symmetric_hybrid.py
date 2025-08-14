# decompress_symmetric_hybrid.py

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torchac

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import AE_hybrid_encoder as AE

import pc_io

def get_cdf_tables(mu, scale, min_val, max_val):
    """
    برای هر بردار فشرده، یک جدول CDF بر اساس μ و σ پیش‌بینی شده می‌سازد.
    """
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
    if not os.path.exists(args.decompressed_path):
        os.makedirs(args.decompressed_path)

    compressed_files_glob = os.path.join(args.compressed_path, '*.h.bin')
    header_files = pc_io.get_files(compressed_files_glob)
    filenames = [os.path.basename(x).replace('.h.bin', '') for x in header_files]
    print(f"Found {len(filenames)} files to decompress.")

    print("Loading symmetric hybrid model...")
    ae_model = AE.get_model(k=args.K, d=args.d).cuda().eval()
    ae_model.load_state_dict(torch.load(args.model_path))
    print("Model loaded successfully.")

    min_val, max_val = -50, 50

    with torch.no_grad():
        for filename in tqdm(filenames, desc="Decompressing"):
            header = np.fromfile(os.path.join(args.compressed_path, filename + '.h.bin'), dtype=np.uint16)
            S, K_from_file, d_from_file = header[0], header[1], header[2]

            assert K_from_file == args.K and d_from_file == args.d, f"Mismatched parameters for {filename}"

            sampled_xyz = np.fromfile(os.path.join(args.compressed_path, filename + '.s.bin'), dtype=np.float16)
            sampled_xyz = torch.from_numpy(sampled_xyz).cuda().reshape(1, S, 3)

            with open(os.path.join(args.compressed_path, filename + '.p.bin'), 'rb') as fin:
                byte_stream = fin.read()

            quantized_features = torch.zeros(S, args.d, device='cuda')
            
            for i in range(S):
                mu, scale = ae_model.bit_estimator(quantized_features)
                cdf_table_i = get_cdf_tables(mu[i:i+1], scale[i:i+1], min_val, max_val)
                symbols_decoded = torchac.decode_float_cdf(cdf_table_i.cpu(), byte_stream)
                quantized_features[i] = (symbols_decoded.cuda().float() + min_val).squeeze(0)
            
            reconstructed_patches_transposed = ae_model.decoder(quantized_features)
            reconstructed_patches = reconstructed_patches_transposed.transpose(1, 2)

            final_pc = (reconstructed_patches + sampled_xyz.squeeze(0).view(S, 1, 3)).reshape(-1, 3)

            output_filepath = os.path.join(args.decompressed_path, filename + '.dec.ply')
            pc_io.save_pc(final_pc.cpu().numpy(), os.path.basename(output_filepath), path=os.path.dirname(output_filepath))

    print("\nDecompression finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='decompress_symmetric_hybrid.py',
        description='Decompress Point Clouds using the trained Symmetric Hybrid Autoencoder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('model_path', help='Path to the trained model state_dict (.pth file).')
    parser.add_argument('compressed_path', help='Directory where compressed files are stored.')
    parser.add_argument('decompressed_path', help='Directory to save decompressed point clouds.')
    parser.add_argument('--K', type=int, help='Number of points in each patch (must match training).', default=1024)
    parser.add_argument('--d', type=int, help='Bottleneck size (must match training).', default=32)

    args = parser.parse_args()
    main(args)