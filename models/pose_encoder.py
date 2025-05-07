# models/pose_encoder.py

import torch
import torch.nn as nn

class PoseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = 72  # 그대로 전달

    def forward(self, pose, num_points):
        # pose: B x 72
        # num_points: int or tensor (B,) → number of points to expand to
        # output: B x N x 72

        if isinstance(num_points, int):
            N = num_points
        else:
            N = num_points[0]  # assume all same for batch

        pose = pose.unsqueeze(1).repeat(1, N, 1)
        return pose
