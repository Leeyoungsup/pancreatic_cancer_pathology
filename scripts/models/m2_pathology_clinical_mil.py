from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from scripts.models.m1_pathology_mil import GatedAttentionMIL, LatentRiskTopKPooling


@dataclass
class M2ModelConfig:
    feature_dim: int = 1024
    coord_dim: int = 6
    clinical_dim: int = 3
    spatial_dim: int = 128
    clinical_embed_dim: int = 64
    fusion_dim: int = 512
    mil_hidden_dim: int = 256
    n_outputs: int = 4
    dropout: float = 0.25
    max_tiles: int = 256
    feature_batch_size: int = 32
    freeze_feature_extractor: bool = True
    use_spatial_embedding: bool = True
    pooling_mode: str = "attention"


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


class ClinicalEmbedding(nn.Module):
    def __init__(self, clinical_dim: int = 3, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(clinical_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, clinical_features: torch.Tensor) -> torch.Tensor:
        if clinical_features.ndim == 1:
            clinical_features = clinical_features.unsqueeze(0)
        return self.net(clinical_features)


class PathologyClinicalMIL(nn.Module):
    def __init__(self, feature_extractor: nn.Module, config: M2ModelConfig):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.config = config

        if config.freeze_feature_extractor:
            self.feature_extractor.eval()
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

        if config.pooling_mode not in {"attention", "mean", "risk_topk"}:
            raise ValueError(
                "pooling_mode must be 'attention', 'mean', or 'risk_topk', "
                f"got {config.pooling_mode!r}"
            )

        self.spatial_embedding = (
            SpatialEmbedding(config.coord_dim, config.spatial_dim, config.dropout)
            if config.use_spatial_embedding
            else None
        )
        self.clinical_embedding = ClinicalEmbedding(config.clinical_dim, config.clinical_embed_dim, config.dropout)
        fusion_input_dim = config.feature_dim + (config.spatial_dim if config.use_spatial_embedding else 0)
        self.tile_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.mil = (
            GatedAttentionMIL(config.fusion_dim, config.mil_hidden_dim, config.dropout)
            if config.pooling_mode == "attention"
            else None
        )
        self.risk_pooling = (
            LatentRiskTopKPooling(config.fusion_dim, config.mil_hidden_dim, config.dropout)
            if config.pooling_mode == "risk_topk"
            else None
        )
        pathology_slide_dim = (
            self.risk_pooling.output_dim
            if self.risk_pooling is not None
            else config.fusion_dim
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(pathology_slide_dim + config.clinical_embed_dim),
            nn.Dropout(config.dropout),
            nn.Linear(pathology_slide_dim + config.clinical_embed_dim, config.n_outputs),
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

    def forward(
        self,
        tile_images: torch.Tensor,
        coords: torch.Tensor,
        clinical_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image_features = self.extract_tile_features(tile_images)
        if self.spatial_embedding is not None:
            spatial_features = self.spatial_embedding(coords.to(image_features.device))
            tile_input = torch.cat([image_features, spatial_features], dim=-1)
        else:
            tile_input = image_features
        fused_tiles = self.tile_fusion(tile_input)
        tile_risk_score = None
        risk_stats = None
        if self.risk_pooling is not None:
            slide_feature, attention, tile_risk_score, risk_stats = self.risk_pooling(fused_tiles)
        elif self.mil is None:
            slide_feature = fused_tiles.mean(dim=0)
            attention = torch.full(
                (fused_tiles.shape[0],),
                fill_value=1.0 / max(fused_tiles.shape[0], 1),
                dtype=fused_tiles.dtype,
                device=fused_tiles.device,
            )
        else:
            slide_feature, attention = self.mil(fused_tiles)

        clinical_embedding = self.clinical_embedding(clinical_features.to(image_features.device)).squeeze(0)
        fused_slide = torch.cat([slide_feature, clinical_embedding], dim=-1)
        logits = self.classifier(fused_slide).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        cumulative_risk = 1.0 - torch.cumprod(1.0 - hazards, dim=-1)
        return {
            "logits": logits,
            "hazard_percent": hazards * 100.0,
            "risk_percent": cumulative_risk * 100.0,
            "attention": attention,
            "slide_feature": slide_feature,
            "tile_risk_score": tile_risk_score,
            "risk_stats": risk_stats,
            "clinical_embedding": clinical_embedding,
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
    return (raw_loss * masks).sum() / masks.sum().clamp_min(1.0)


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


def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-4) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
