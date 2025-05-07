# test.py

import torch
import numpy as np
from models.segmentation_net import SegmentationNet
from datasets.close_dataset import CloSeDataset, close_collate
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

def load_model(ckpt_path, device='cuda'):
    model = SegmentationNet(num_classes=18)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, loader, device='cuda', save_dir='./outputs'):
    os.makedirs(save_dir, exist_ok=True)

    for batch in tqdm(loader):
        points = batch['points'].to(device)                   # B x N x 3
        scalar = batch['scalar_features'].to(device)          # B x N x 6
        pose = batch['pose'].to(device)                       # B x 72
        garments = batch['garments'].to(device)               # B x 18

        logits = model(points, scalar, pose, garments)        # B x N x C
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy() # (N,)

        # Save prediction (can also save .ply/.obj/.npz if needed)
        fname = os.path.basename(loader.dataset.files[0]).replace('.npz', '_pred.npy')
        np.save(os.path.join(save_dir, fname), pred)

        print(f"Saved prediction to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to trained model (.pt or .pth)')
    parser.add_argument('--input', type=str, required=True, help='Path to test npz file or directory')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    test_files = []
    if os.path.isdir(args.input):
        test_files = sorted([os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith('.npz')])
    else:
        test_files = [args.input]

    test_dataset = CloSeDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=close_collate)

    model = load_model(args.ckpt, args.device)
    run_inference(model, test_loader, device=args.device)
