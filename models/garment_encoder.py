# models/garment_encoder.py

import torch
import torch.nn as nn

class GarmentEncoder(nn.Module):
    def __init__(self, in_dim=18, hidden_dim=32, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, garments, num_points):
        # garments: B x 18
        # return: B x N x out_dim → expand across all points

        feat = self.mlp(garments)           # B x out_dim
        feat = feat.unsqueeze(1).repeat(1, num_points, 1)  # B x N x out_dim
        return feat
