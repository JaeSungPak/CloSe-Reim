# models/point_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointEncoder(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, points, scalar_features):
        # points: B x N x 3
        # scalar_features: B x N x 6
        x = torch.cat([points, scalar_features], dim=-1)  # B x N x 9
        x = self.mlp(x)  # B x N x out_dim
        return x
