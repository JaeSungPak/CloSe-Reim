# models/segmentation_net.py

import torch
import torch.nn as nn
from models.point_encoder import PointEncoder
from models.pose_encoder import PoseEncoder
from models.garment_encoder import GarmentEncoder

class SegmentationNet(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        self.point_enc = PointEncoder(in_dim=9, hidden_dim=64, out_dim=128)
        self.pose_enc = PoseEncoder()
        self.garm_enc = GarmentEncoder(in_dim=18, hidden_dim=32, out_dim=64)

        total_feat_dim = 128 + 72 + 64  # Point + Pose + Garment
        self.decoder = nn.Sequential(
            nn.Linear(total_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, points, scalar_features, pose, garments):
        B, N, _ = points.shape

        pt_feat = self.point_enc(points, scalar_features)         # B x N x 128
        pose_feat = self.pose_enc(pose, N)                        # B x N x 72
        garm_feat = self.garm_enc(garments, N)                    # B x N x 64

        fused = torch.cat([pt_feat, pose_feat, garm_feat], dim=-1)  # B x N x (128+72+64)
        logits = self.decoder(fused)  # B x N x num_classes

        return logits
