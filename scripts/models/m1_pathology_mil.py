from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class M1ModelConfig:
    feature_dim: int = 1024
    coord_dim: int = 6
    spatial_dim: int = 128
    fusion_dim: int = 512
    mil_hidden_dim: int = 256
    n_outputs: int = 4
    dropout: float = 0.25
    max_tiles: int = 256
    feature_batch_size: int = 32
    freeze_feature_extractor: bool = True


class SpatialEmbedding(nn.Module):
    def __init__(self, coord_dim: int = 6, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coord_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


class GatedAttentionMIL(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, tile_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention = self.attention_w(self.attention_v(tile_features) * self.attention_u(tile_features))
        attention = torch.softmax(attention.squeeze(-1), dim=0)
        bag_feature = torch.sum(tile_features * attention.unsqueeze(-1), dim=0)
        return bag_feature, attention


class PathologySpatialMIL(nn.Module):
    def __init__(self, feature_extractor: nn.Module, config: M1ModelConfig):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.config = config

        if config.freeze_feature_extractor:
            self.feature_extractor.eval()
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        self.spatial_embedding = SpatialEmbedding(
            coord_dim=config.coord_dim,
            embed_dim=config.spatial_dim,
            dropout=config.dropout,
        )
        self.tile_fusion = nn.Sequential(
            nn.Linear(config.feature_dim + config.spatial_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mil = GatedAttentionMIL(
            input_dim=config.fusion_dim,
            hidden_dim=config.mil_hidden_dim,
            dropout=config.dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.n_outputs),
        )

    @staticmethod
    def _unwrap_features(output: Any) -> torch.Tensor:
        if isinstance(output, dict):
            for key in ["x_norm_clstoken", "pooler_output", "last_hidden_state"]:
                if key in output:
                    output = output[key]
                    break
            else:
                output = next(iter(output.values()))
        elif isinstance(output, (tuple, list)):
            output = output[0]

        if output.ndim == 3:
            output = output[:, 0]
        if output.ndim > 2:
            output = torch.flatten(output, start_dim=1)
        return output

    def extract_tile_features(self, tile_images: torch.Tensor) -> torch.Tensor:
        features = []
        context = torch.no_grad() if self.config.freeze_feature_extractor else torch.enable_grad()
        with context:
            for tile_batch in tile_images.split(self.config.feature_batch_size, dim=0):
                feature = self.feature_extractor(tile_batch)
                feature = self._unwrap_features(feature)
                features.append(feature)
        return torch.cat(features, dim=0)

    def forward(self, tile_images: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        if tile_images.ndim != 4:
            raise ValueError(f"tile_images must be [N, C, H, W], got {tuple(tile_images.shape)}")
        if coords.ndim != 2:
            raise ValueError(f"coords must be [N, coord_dim], got {tuple(coords.shape)}")
        if tile_images.shape[0] != coords.shape[0]:
            raise ValueError("tile_images and coords must have the same number of tiles.")

        image_features = self.extract_tile_features(tile_images)
        spatial_features = self.spatial_embedding(coords.to(image_features.device))
        fused_tiles = self.tile_fusion(torch.cat([image_features, spatial_features], dim=-1))
        slide_feature, attention = self.mil(fused_tiles)
        logits = self.classifier(slide_feature).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        return {
            "logits": logits,
            "hazard_percent": hazards * 100.0,
            "risk_percent": cumulative_risk * 100.0,
            "attention": attention,
            "slide_feature": slide_feature,
        }


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    raw_loss = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    masks = masks.to(raw_loss.device).float()
    denom = masks.sum().clamp_min(1.0)
    return (raw_loss * masks).sum() / denom


def sample_tiles(
    tile_paths: list[str],
    coords: torch.Tensor,
    max_tiles: int,
    training: bool = True,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    n_tiles = len(tile_paths)
    if n_tiles == 0:
        raise ValueError("tile_paths is empty.")

    if n_tiles <= max_tiles:
        indices = torch.arange(n_tiles)
    elif training:
        indices = torch.randperm(n_tiles)[:max_tiles]
    else:
        indices = torch.linspace(0, n_tiles - 1, steps=max_tiles).long()

    selected_paths = [tile_paths[int(i)] for i in indices]
    return selected_paths, coords[indices], indices


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
