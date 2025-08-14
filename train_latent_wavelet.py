# train_latent_wavelet.py

import os, sys, argparse, torch, numpy as np
from tqdm import tqdm
import torch.utils.data as Data
from pytorch3d.ops import knn_points

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import AE_latent_wavelet as AE, pc_kit, pc_io

torch.manual_seed(1); torch.cuda.manual_seed(1); np.random.seed(1)

def main(args):
    os.makedirs(args.model_save_folder, exist_ok=True)
    

    print("--- Loading and preparing data... ---")
    p_min, p_max, _ = pc_io.get_shape_data(1)
    files = pc_io.get_files(args.train_glob)
    loaded_points_list = pc_io.load_points(files, p_min, p_max, processbar=True)
    points = torch.from_numpy(np.stack(loaded_points_list).astype(np.float32))    
    S, K = args.N * args.ALPHA // args.K, args.K
    
    patches_list = [
        (knn_points(pc_kit.index_points(pc.unsqueeze(0), pc_kit.farthest_point_sample_batch(pc.unsqueeze(0), S)), 
                    pc.unsqueeze(0), K=K, return_nn=True)[2] - 
         pc_kit.index_points(pc.unsqueeze(0), pc_kit.farthest_point_sample_batch(pc.unsqueeze(0), S)).view(1, S, 1, 3)
        ).view(-1, K, 3) 
        for pc in tqdm(points, desc="Creating Patches")
    ]
    patches = torch.cat(patches_list, dim=0)
    print(f"Created {patches.shape[0]} patches of size {patches.shape[1:]}")

    loader = Data.DataLoader(patches, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    k_out = args.K // args.ALPHA
    ae_model = AE.get_model(k=k_out, d=args.d).cuda().train()
    criterion = AE.get_loss().cuda()
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay)
    
    print(f"--- Starting Training ---")
    print(f"Parameters: K={args.K}, d={args.d}, lambda={args.lamda}")
    
    global_step = 0
    for epoch in range(9999):
        for batch_x in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch_x_cuda = batch_x.cuda(non_blocking=True)
            batch_x_pred, bpp = ae_model(batch_x_cuda)
            
            current_lamda = args.lamda if global_step >= args.rate_loss_enable_step else 0
            loss = criterion(batch_x_pred, batch_x_cuda, bpp, current_lamda)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            
            if global_step % 500 == 0:
                tqdm.write(f"\nStep: {global_step} | Loss: {loss.item():.5f} | BPP: {bpp.item():.5f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                torch.save(ae_model.state_dict(), os.path.join(args.model_save_folder, 'ae_latent_wavelet.pth'))
            
            if global_step >= args.max_steps: break
        if global_step >= args.max_steps: break
        
    print("--- Training finished successfully. ---")
    torch.save(ae_model.state_dict(), os.path.join(args.model_save_folder, 'ae_latent_wavelet_final.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('train_glob')
    parser.add_argument('model_save_folder')
    parser.add_argument('--N', type=int, default=8192)
    parser.add_argument('--ALPHA', type=int, default=2)
    parser.add_argument('--K', type=int, default=1024)
    parser.add_argument('--d', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lamda', type=float, default=1e-7)
    parser.add_argument('--rate_loss_enable_step', type=int, default=40000)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--max_steps', type=int, default=100000)
    main(parser.parse_args())
