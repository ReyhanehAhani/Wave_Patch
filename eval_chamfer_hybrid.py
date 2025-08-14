# eval_chamfer_hybrid.py (نسخه نهایی با اصلاح مسیر هوشمند)

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from pytorch3d.loss import chamfer_distance

# مسیر پوشه والد را اضافه می‌کنیم تا ماژول pc_io پیدا شود
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pc_io

def get_file_size_in_bits(filepath):
    return os.stat(filepath).st_size * 8

def main(args):
    decompressed_files_glob = os.path.join(args.decompressed_path, '*.dec.ply')
    decompressed_files = pc_io.get_files(decompressed_files_glob)
    print(f"Found {len(decompressed_files)} decompressed files to evaluate.")
    
    results = []

    for decomp_file_path in tqdm(decompressed_files, desc="Evaluating Chamfer Distance & BPP"):
        filename_base = os.path.basename(decomp_file_path).replace('.dec.ply', '')
        
       
        category = filename_base.rsplit('_', 1)[0]
        original_file_path = os.path.join(args.original_path, category, 'test', filename_base + '.ply')

        if not os.path.exists(original_file_path):
            print(f"Warning: Original file not found at '{original_file_path}', skipping.")
            continue

        original_points_np = pc_io.load_points([original_file_path], 0, 1, processbar=False)[0]
        n_points_original = original_points_np.shape[0]
        
        comp_p_file = os.path.join(args.compressed_path, filename_base + '.p.bin')
        comp_s_file = os.path.join(args.compressed_path, filename_base + '.s.bin')
        comp_h_file = os.path.join(args.compressed_path, filename_base + '.h.bin')
        total_bits = (get_file_size_in_bits(comp_p_file) +
                      get_file_size_in_bits(comp_s_file) +
                      get_file_size_in_bits(comp_h_file))
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
        print("\nError: No files were evaluated. Please check your input paths.")
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
        prog='eval_chamfer_hybrid.py',
        description='Evaluate BPP and Chamfer Distance for the Hybrid model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('original_path', help='Base directory of the original test point clouds (e.g., .../ModelNet40_pc_01_8192p).')
    parser.add_argument('compressed_path', help='Directory of the compressed files.')
    parser.add_argument('decompressed_path', help='Directory of the decompressed point clouds.')
    parser.add_argument('output_csv', help='Path to save the output CSV file with results.')
    
    args = parser.parse_args()
    main(args)