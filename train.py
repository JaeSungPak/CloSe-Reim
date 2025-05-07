# train.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import glob
from tqdm import tqdm

from datasets.close_dataset import CloSeDataset, close_collate
from models.segmentation_net import SegmentationNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------- Config -----------

data_root = './data/train/*.npz'
batch_size = 1
num_epochs = 20
lr = 1e-3
num_classes = 18

# ----------- Data -----------

file_list = sorted(glob.glob(data_root))
dataset = CloSeDataset(file_list)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=close_collate)

# ----------- Model -----------

model = SegmentationNet(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ----------- Train Loop -----------

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        points = batch['points'].to(device)                   # B x N x 3
        scalar = batch['scalar_features'].to(device)          # B x N x 6
        pose = batch['pose'].to(device)                       # B x 72
        garments = batch['garments'].to(device)               # B x 18
        labels = batch['labels'].to(device)                   # B x N

        logits = model(points, scalar, pose, garments)        # B x N x C
        loss = criterion(logits.view(-1, num_classes), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
