# train_hybrid_encoder.py

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as Data
from pytorch3d.ops.knn import knn_points


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import AE_hybrid_encoder as AE
import pc_kit
import pc_io

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def main(args):

    if not os.path.exists(args.model_save_folder):
        os.makedirs(args.model_save_folder)
        print(f"Created model save directory: {args.model_save_folder}")

    print("--- Data Preparation & Patch Division ---")
    p_min, p_max, _ = pc_io.get_shape_data(1)
    files = pc_io.get_files(args.train_glob)
    print(f"Found {len(files)} files for training.")
    
    all_patches = []
    S = args.N * args.ALPHA // args.K
    
    for f in tqdm(files, desc="Loading files and creating patches"):
        points_np = pc_io.load_point_cloud(f)
        if points_np is None:
            continue
            
        pc = torch.from_numpy(points_np.astype(np.float32)).unsqueeze(0)
        
        sampled_centers = pc_kit.index_points(pc, pc_kit.farthest_point_sample_batch(pc, S))
        _, _, grouped_xyz = knn_points(sampled_centers, pc, K=args.K, return_nn=True)
        grouped_xyz -= sampled_centers.view(1, S, 1, 3)
        
        all_patches.append(grouped_xyz.view(-1, args.K, 3))

    patches = torch.cat(all_patches, dim=0)
    print(f"Total patches created: {patches.shape}")

    loader = Data.DataLoader(
        dataset=patches,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("\n--- Model Setup ---")
    ae_model = AE.get_model(k=args.K, d=args.d).cuda().train()
    criterion = AE.get_loss().cuda()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay)

    print(f"Model created. Training with K={args.K}, d={args.d}, lambda={args.lamda}")

    print("\n--- Starting Training ---")
    global_step = 0
    bpps, losses = [], []
    training_complete = False

    for epoch in range(9999):
        for batch_x in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch_x = batch_x.cuda(non_blocking=True)
            
            batch_x_pred, bpp = ae_model(batch_x)
            
            current_lamda = 0 if global_step < args.rate_loss_enable_step else args.lamda
            loss = criterion(batch_x, batch_x_pred, bpp, current_lamda)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            losses.append(loss.item())
            bpps.append(bpp.item())

            if global_step % 500 == 0:
                avg_loss = np.mean(losses)
                avg_bpp = np.mean(bpps)
                print(f"\nEpoch: {epoch+1} | Step: {global_step} | Est. BPP: {avg_bpp:.5f} | Loss: {avg_loss:.5f} | Lambda: {current_lamda}")
                losses, bpps = [], []
                
                torch.save(ae_model.state_dict(), os.path.join(args.model_save_folder, 'ae_symmetric_decoder.pth'))

            scheduler.step()
            
            if global_step >= args.max_steps:
                training_complete = True
                break
        
        if training_complete:
            break
            
    print("\n--- Training Finished! ---")
    final_model_path = os.path.join(args.model_save_folder, 'ae_symmetric_decoder_final.pth')
    torch.save(ae_model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train_hybrid_encoder.py',
        description='Train the new Hybrid Autoencoder with a Symmetric Decoder.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('train_glob', help='Point clouds glob pattern for training.')
    parser.add_argument('model_save_folder', help='Directory where to save trained models.')
    parser.add_argument('--N', type=int, help='Total points in the original point cloud.', default=8192)
    parser.add_argument('--ALPHA', type=int, help='The factor of patch coverage ratio.', default=2)
    parser.add_argument('--K', type=int, help='Number of points in each patch.', default=1024)
    parser.add_argument('--d', type=int, help='Bottleneck size (latent dimension).', default=32)
    parser.add_argument('--lr', type=float, help='Initial learning rate.', default=1e-4)
    parser.add_argument('--batch_size', type=int, help='Number of patches in a batch.', default=16)
    parser.add_argument('--lamda', type=float, help='Lambda for rate-distortion tradeoff.', default=1e-6)
    parser.add_argument('--rate_loss_enable_step', type=int, help='Apply rate-distortion loss after this many steps.', default=40000)
    parser.add_argument('--lr_decay', type=float, help='Learning rate decay factor.', default=0.5)
    parser.add_argument('--lr_decay_steps', type=int, help='Decay learning rate every N steps.', default=50000)
    parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=100000)

    args = parser.parse_args()
    main(args)