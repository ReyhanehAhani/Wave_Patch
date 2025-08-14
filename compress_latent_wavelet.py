import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torchac

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import AE_latent_wavelet as AE
import pc_kit
import pc_io

def pmf_to_cdf(pmf):
  
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0

def main(args):
    
    if not os.path.exists(args.compressed_path):
        os.makedirs(args.compressed_path)
        print(f"Created directory: {args.compressed_path}")

    try:
        model_file = os.path.join(args.model_path, 'ae_latent_wavelet_final.pth')
        k_out = args.K // args.ALPHA
        ae_model = AE.get_model(k=k_out, d=args.d).cuda().eval()
        ae_model.load_state_dict(torch.load(model_file))
        print(f"Model loaded successfully from {model_file}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_file}")
        sys.exit(1)

    files = pc_io.get_files(args.uncompressed_glob)
    if len(files) == 0:
        print(f"Error: No input files found with glob pattern: {args.uncompressed_glob}")
        sys.exit(1)

    print(f"Found {len(files)} point clouds to compress.")
    print("Generating CDF tables from trained model...")
    L = 255
    pmf_lf = ae_model.be_lf.get_pmf(device='cuda', L=L)
    cdf_lf = pmf_to_cdf(pmf_lf).cpu()  

    pmf_hf = ae_model.be_hf.get_pmf(device='cuda', L=L)
    cdf_hf = pmf_to_cdf(pmf_hf).cpu() 
    print("CDF tables created.")

    processed_count = 0
    skipped_count = 0

    with torch.no_grad():
        for f_path in tqdm(files, desc="Compressing Point Clouds"):
            try:
                points_np = pc_io.load_point_cloud(f_path)
                if points_np is None:
                    tqdm.write(f"Skipping file {os.path.basename(f_path)}: Could not be read.")
                    skipped_count += 1
                    continue

                num_points = points_np.shape[0]
                if num_points < args.K:
                    tqdm.write(f"Skipping file {os.path.basename(f_path)}: not enough points ({num_points} < K={args.K}).")
                    skipped_count += 1
                    continue

                S = (num_points * args.ALPHA) // args.K
                if S == 0:
                    tqdm.write(f"Skipping file {os.path.basename(f_path)}: not enough points to create even one patch.")
                    skipped_count += 1
                    continue

                pc = torch.from_numpy(points_np).cuda().float().unsqueeze(0)
                patches_xyz = pc_kit.index_points(pc, pc_kit.farthest_point_sample_batch(pc, S))
                patches = (pc_kit.knn_points(patches_xyz, pc, K=args.K, return_nn=True)[2] - patches_xyz.view(1, S, 1, 3)).squeeze(0)

                feature = ae_model.sa(patches.transpose(1, 2))
                feature = ae_model.pn(torch.cat((patches.transpose(1, 2), feature), dim=1))
                feature_reshaped = feature.unsqueeze(1)
                yl, yh_list = ae_model.dwt(feature_reshaped)
                yh = yh_list[0]

                q_yl = torch.round(yl).to(torch.int16)
                q_yh = torch.round(yh).to(torch.int16)

                base_filename = os.path.splitext(os.path.basename(f_path))[0]

                symbols_lf = (q_yl.squeeze(1) + (L // 2)).cpu()
                byte_stream_lf = torchac.encode_float_cdf(cdf_lf.repeat(S, 1, 1), symbols_lf, check_input_bounds=True)
                with open(os.path.join(args.compressed_path, f'{base_filename}.lf.bin'), 'wb') as fout:
                    fout.write(byte_stream_lf)

                symbols_hf = (q_yh.squeeze(1) + (L // 2)).cpu()
                byte_stream_hf = torchac.encode_float_cdf(cdf_hf.repeat(S, 1, 1), symbols_hf, check_input_bounds=True)
                with open(os.path.join(args.compressed_path, f'{base_filename}.hf.bin'), 'wb') as fout:
                    fout.write(byte_stream_hf)

                centroids_to_save = (patches_xyz.squeeze(0).cpu().numpy()).astype(np.float16)
                centroids_to_save.tofile(os.path.join(args.compressed_path, f'{base_filename}.s.bin'))
                np.array([centroids_to_save.shape[0]], dtype=np.uint16).tofile(os.path.join(args.compressed_path, f'{base_filename}.h.bin'))

                processed_count += 1

            except Exception as e:
                tqdm.write(f"An unexpected error occurred processing file {f_path}: {e}")
                skipped_count += 1
                continue

    print("\n--- Compression process finished. ---")
    print(f"Successfully processed: {processed_count} files.")
    print(f"Skipped: {skipped_count} files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_path')
    parser.add_argument('uncompressed_glob')
    parser.add_argument('compressed_path')
    parser.add_argument('--K', type=int, default=1024)
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--ALPHA', type=int, default=2)
    args = parser.parse_args()
    main(args)
