from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scripts.models.m1_pathology_mil import GatedAttentionMIL, LatentRiskTopKPooling, PathologySpatialMIL, SpatialEmbedding
from scripts.models.m2_pathology_clinical_mil import ClinicalEmbedding
from scripts.models.m3_pathology_rnaseq_mil import RNASeqEmbedding, build_optimizer, sample_tiles


@dataclass
class M4ModelConfig:
    feature_dim: int = 1024
    coord_dim: int = 6
    clinical_dim: int = 3
    rnaseq_dim: int = 1500
    spatial_dim: int = 128
    clinical_embed_dim: int = 16
    rnaseq_hidden_dim: int = 256
    rnaseq_embed_dim: int = 64
    fusion_dim: int = 128
    mil_hidden_dim: int = 64
    n_outputs: int = 1
    dropout: float = 0.5
    rnaseq_dropout: float = 0.5
    max_tiles: int = 512
    feature_batch_size: int = 64
    freeze_feature_extractor: bool = True
    use_spatial_embedding: bool = False
    pooling_mode: str = "risk_topk"


class PathologyRNASeqClinicalMIL(PathologySpatialMIL):
    def __init__(self, feature_extractor: nn.Module, config: M4ModelConfig):
        nn.Module.__init__(self)
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
        self.rnaseq_embedding = RNASeqEmbedding(
            input_dim=config.rnaseq_dim,
            hidden_dim=config.rnaseq_hidden_dim,
            embed_dim=config.rnaseq_embed_dim,
            dropout=config.rnaseq_dropout,
        )

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
        pathology_slide_dim = self.risk_pooling.output_dim if self.risk_pooling is not None else config.fusion_dim
        classifier_input_dim = pathology_slide_dim + config.rnaseq_embed_dim + config.clinical_embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_dim),
            nn.Dropout(config.dropout),
            nn.Linear(classifier_input_dim, config.n_outputs),
        )

    def forward(
        self,
        tile_images: torch.Tensor,
        coords: torch.Tensor,
        rnaseq_features: torch.Tensor,
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

        rnaseq_embedding = self.rnaseq_embedding(rnaseq_features.to(image_features.device)).squeeze(0)
        clinical_embedding = self.clinical_embedding(clinical_features.to(image_features.device)).squeeze(0)
        fused_slide = torch.cat([slide_feature, rnaseq_embedding, clinical_embedding], dim=-1)
        logits = self.classifier(fused_slide).unsqueeze(0)
        hazards = torch.sigmoid(logits)
        return {
            "logits": logits,
            "hazard_percent": hazards * 100.0,
            "risk_percent": hazards * 100.0,
            "attention": attention,
            "slide_feature": slide_feature,
            "tile_risk_score": tile_risk_score,
            "risk_stats": risk_stats,
            "rnaseq_embedding": rnaseq_embedding,
            "clinical_embedding": clinical_embedding,
        }


__all__ = [
    "M4ModelConfig",
    "PathologyRNASeqClinicalMIL",
    "build_optimizer",
    "sample_tiles",
]
