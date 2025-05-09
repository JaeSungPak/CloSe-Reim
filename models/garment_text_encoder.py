import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class GarmentTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", proj_dim=64, device='cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        self.text_dim = self.text_encoder.config.hidden_size  # usually 512

        self.project = nn.Sequential(
            nn.Linear(self.text_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encodes a single text prompt into a (D,) tensor."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.text_encoder(**inputs)
            text_feat = output.last_hidden_state[:, 0, :]  # CLS token
            projected = self.project(text_feat)  # (1, D)
        return projected.squeeze(0).cpu()  # return to CPU for caching

    def forward(self, text_list: list[str], num_points: int) -> torch.Tensor:
        """Encodes a list of prompts into (B, N, D) tensor."""
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
        output = self.text_encoder(**inputs)
        text_feat = output.last_hidden_state[:, 0, :]  # (B, 512)
        projected = self.project(text_feat)  # (B, D)
        return projected.unsqueeze(1).repeat(1, num_points, 1)  # (B, N, D)
