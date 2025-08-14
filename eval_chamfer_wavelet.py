# eval_chamfer_wavelet.py

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from pytorch3d.loss import chamfer_distance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pc_io

def get_file_size_in_bits(filepath):
    return os.stat(filepath).st_size * 8

def main(args):
    decompressed_files_glob = os.path.join(args.decompressed_path, '*_rec.ply')
    decompressed_files = pc_io.get_files(decompressed_files_glob)
    print(f"Found {len(decompressed_files)} decompressed files to evaluate.")

    results = []

    for decomp_file_path in tqdm(decompressed_files, desc="Evaluating Chamfer Distance & BPP"):
        filename_base = os.path.basename(decomp_file_path).replace('_rec.ply', '')
        category = filename_base.rsplit('_', 1)[0]
        original_file_path = os.path.join(args.original_path, category, 'test', filename_base + '.ply')

        if not os.path.exists(original_file_path):
            print(f"Warning: Original file not found at '{original_file_path}', skipping.")
            continue

        comp_lf_file = os.path.join(args.compressed_path, filename_base + '.lf.bin')
        comp_hf_file = os.path.join(args.compressed_path, filename_base + '.hf.bin')
        comp_h_file  = os.path.join(args.compressed_path, filename_base + '.h.bin')
        comp_s_file  = os.path.join(args.compressed_path, filename_base + '.s.bin')

        required_files = [comp_lf_file, comp_hf_file, comp_h_file, comp_s_file]
        if not all(os.path.exists(f) for f in required_files):
            print(f"Warning: Missing one or more compressed files for {filename_base}, skipping.")
            continue

        original_points_np = pc_io.load_points([original_file_path], 0, 1, processbar=False)[0]
        n_points_original = original_points_np.shape[0]

        total_bits = (get_file_size_in_bits(comp_lf_file) +
                      get_file_size_in_bits(comp_hf_file) +
                      get_file_size_in_bits(comp_h_file) +
                      get_file_size_in_bits(comp_s_file))
        bpp = total_bits / n_points_original

        decomp_points_np = pc_io.load_points([decomp_file_path], 0, 1, processbar=False)[0]
        original_pc_tensor = torch.from_numpy(original_points_np).float().cuda().unsqueeze(0)
        decomp_pc_tensor = torch.from_numpy(decomp_points_np).float().cuda().unsqueeze(0)

        chamfer_dist, _ = chamfer_distance(original_pc_tensor, decomp_pc_tensor)

        results.append({
            'filename': filename_base,
            'chamfer_distance': chamfer_dist.item(),
            'bpp': bpp
        })

    if not results:
        print("\nError: No valid files were evaluated. Please check file paths and compressed files.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nEvaluation results saved to {args.output_csv}")

    avg_chamfer = df['chamfer_distance'].mean()
    avg_bpp = df['bpp'].mean()

    print("\n--- Average Results ---")
    print(f"Chamfer Distance: {avg_chamfer:.8f}")
    print(f"Bits Per Point (BPP): {avg_bpp:.4f}")
    print("-----------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='eval_chamfer_wavelet.py',
        description='Evaluate BPP and Chamfer Distance for Wavelet-based model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('original_path', help='Directory of original test point clouds (e.g., .../ModelNet40_pc_01_8192p).')
    parser.add_argument('compressed_path', help='Directory of compressed files.')
    parser.add_argument('decompressed_path', help='Directory of decompressed point clouds.')
    parser.add_argument('output_csv', help='Path to save evaluation results CSV.')
    
    args = parser.parse_args()
    main(args)