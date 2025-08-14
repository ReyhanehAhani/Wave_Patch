import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import torchac

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import AE_latent_wavelet as AE
import pc_kit
import pc_io

def main(args):
    if not os.path.exists(args.reconstructed_path):
        os.makedirs(args.reconstructed_path)
        print(f"Created output directory: {args.reconstructed_path}")

    try:
        model_file = os.path.join(args.model_path, 'ae_latent_wavelet_final.pth')
        k_out = args.K // args.ALPHA
        ae_model = AE.get_model(k=k_out, d=args.d).cuda().eval()
        ae_model.load_state_dict(torch.load(model_file))
        print(f"Model loaded successfully from {model_file}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_file}")
        sys.exit(1)

    print("Generating CDF tables from trained model...")
    L = 255
    pmf_lf = ae_model.be_lf.get_pmf(device='cuda', L=L)
    cdf_lf = pmf_lf.cumsum(dim=-1)
    cdf_lf = torch.cat([torch.zeros_like(cdf_lf[..., :1]), cdf_lf], dim=-1).clamp(max=1.).cpu()

    pmf_hf = ae_model.be_hf.get_pmf(device='cuda', L=L)
    cdf_hf = pmf_hf.cumsum(dim=-1)
    cdf_hf = torch.cat([torch.zeros_like(cdf_hf[..., :1]), cdf_hf], dim=-1).clamp(max=1.).cpu()

    bin_files = [f for f in os.listdir(args.compressed_path) if f.endswith('.lf.bin')]
    print(f"Found {len(bin_files)} compressed samples.")

    for lf_file in tqdm(bin_files, desc="Decompressing"):
        base = lf_file.replace('.lf.bin', '')
        try:
            lf_path = os.path.join(args.compressed_path, f'{base}.lf.bin')
            hf_path = os.path.join(args.compressed_path, f'{base}.hf.bin')
            s_path  = os.path.join(args.compressed_path, f'{base}.s.bin')
            h_path  = os.path.join(args.compressed_path, f'{base}.h.bin')

            if not all(os.path.exists(p) for p in [lf_path, hf_path, s_path, h_path]):
                tqdm.write(f"Missing some bin files for {base}, skipping.")
                continue

            S = np.fromfile(h_path, dtype=np.uint16)[0]
            centroids = np.fromfile(s_path, dtype=np.float16).reshape(S, 3)

            lf_stream = open(lf_path, 'rb').read()
            hf_stream = open(hf_path, 'rb').read()

            symbols_lf = torchac.decode_float_cdf(cdf_lf.repeat(S, 1, 1), lf_stream)
            symbols_hf = torchac.decode_float_cdf(cdf_hf.repeat(S, 1, 1), hf_stream)

            q_yl = (symbols_lf - (L // 2)).to(torch.float32).cuda().unsqueeze(1)
            q_yh = (symbols_hf - (L // 2)).to(torch.float32).cuda().unsqueeze(1)

            y = ae_model.iwt(q_yl, q_yh)
            recon = ae_model.decoder(y.squeeze(1)).detach().cpu().numpy()

            K_real = recon.shape[1] // 3
            recon = recon.reshape(S, K_real, 3)

            recon_full = []
            for i in range(S):
                patch = recon[i] + centroids[i]
                recon_full.append(patch)
            recon_full = np.concatenate(recon_full, axis=0)

            out_path = os.path.join(args.reconstructed_path, f'{base}_rec.ply')
            pc_io.save_point_cloud(recon_full, out_path)

        except Exception as e:
            tqdm.write(f"Failed to decompress {base}: {e}")
            continue

    print("\n--- Decompression completed ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('compressed_path')
    parser.add_argument('reconstructed_path')
    parser.add_argument('--K', type=int, default=1024)
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--ALPHA', type=int, default=2)
    args = parser.parse_args()
    main(args)
