# M1 모델 정의 및 loss/optimizer 구성
# 구조: frozen UNI/UNI v2 feature extractor + coordinate spatial embedding + attention MIL

import torch
from torch import nn
import pandas as pd
import timm
from huggingface_hub import hf_hub_download

from scripts.models.discrete_survival import (
    cox_ph_loss,
    harrell_c_index,
)

from scripts.models.m1_pathology_mil import (
    M1ModelConfig,
    PathologySpatialMIL,
    sample_tiles,
    build_optimizer,
)

if "device" not in globals():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

MAX_TILES_PER_SLIDE = 512
FEATURE_BATCH_SIZE = 64
SPATIAL_DIM = 128
FUSION_DIM = 128
MIL_HIDDEN_DIM = 64
USE_SPATIAL_EMBEDDING = False
POOLING_MODE = "risk_topk"
DROPOUT = 0.50
N_OUTPUTS = 1  # patient-level Cox risk score
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-3

# "UNI" 또는 "UNI2-h" 중 선택하세요.
# UNI/UNI2-h는 Hugging Face 접근 권한 또는 로컬 캐시가 필요할 수 있습니다.
FEATURE_EXTRACTOR_NAME = "UNI2-h"
UNI_BACKBONES = {
    "UNI": {
        "repo_id": "MahmoodLab/UNI",
        "filename": "pytorch_model.bin",
        "feature_dim": 1024,
        "timm_kwargs": {
            "model_name": "vit_large_patch16_224",
            "img_size": 224,
            "patch_size": 16,
            "init_values": 1e-5,
            "num_classes": 0,
            "dynamic_img_size": True,
        },
    },
    "UNI2-h": {
        "repo_id": "MahmoodLab/UNI2-h",
        "filename": "pytorch_model.bin",
        "feature_dim": 1536,
        "timm_kwargs": {
            "model_name": "vit_giant_patch14_224",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
    },
}


def load_uni_feature_extractor(
    name: str = FEATURE_EXTRACTOR_NAME,
    device: torch.device = device,
    local_files_only: bool = False,
) -> tuple[nn.Module, int]:
    if name not in UNI_BACKBONES:
        raise ValueError(f"지원하지 않는 feature extractor입니다: {name}. choices={list(UNI_BACKBONES)}")

    cfg = UNI_BACKBONES[name]
    print(f"Loading {name}: {cfg['repo_id']} / {cfg['timm_kwargs']['model_name']}")

    model = timm.create_model(
        pretrained=False,
        **cfg["timm_kwargs"],
    )

    try:
        weight_path = hf_hub_download(
            repo_id=cfg["repo_id"],
            filename=cfg["filename"],
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"{name} weight를 Hugging Face에서 가져오지 못했습니다. "
            "접근 권한/로그인 토큰 또는 네트워크/캐시를 확인하세요. "
            f"repo_id={cfg['repo_id']}, filename={cfg['filename']}"
        ) from exc

    state_dict = torch.load(weight_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print("load_state_dict warning")
        print("missing keys:", len(missing))
        print("unexpected keys:", len(unexpected))

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device)

    feature_dim = int(getattr(model, "num_features", cfg["feature_dim"]))
    print(f"{name} loaded. feature_dim={feature_dim}")
    return model, feature_dim


# 셀 3에서 UNI/UNI v2까지 로드하고 바로 M1 model/optimizer를 구성합니다.
feature_extractor, M1_FEATURE_DIM = load_uni_feature_extractor(
    name=FEATURE_EXTRACTOR_NAME,
    device=device,
    local_files_only=False,
)
FEATURE_EXTRACTOR_PATCH_SIZE = int(UNI_BACKBONES[FEATURE_EXTRACTOR_NAME]["timm_kwargs"]["patch_size"])
print("FEATURE_EXTRACTOR_PATCH_SIZE:", FEATURE_EXTRACTOR_PATCH_SIZE)
print("PATCH_INPUT_SIZE:", PATCH_INPUT_SIZE, "-> model input size:", get_model_input_size(PATCH_INPUT_SIZE))
print("PATCH_PADDING:", get_patch_padding(PATCH_INPUT_SIZE))

# 2번 셀에서 만든 tile-level dataset transform은 feature extractor patch size를 알기 전 생성되므로 여기서 갱신합니다.
# resized tile cache가 있으면 512 resize를 반복하지 않고 padding/augmentation/normalization만 수행합니다.
if bool(globals().get("TILE_IMAGE_CACHE", {})):
    train_tile_dataset.transform = get_train_cached_patch_transform()
    valid_tile_dataset.transform = get_eval_cached_patch_transform()
    test_tile_dataset.transform = get_eval_cached_patch_transform()
else:
    train_tile_dataset.transform = get_train_patch_transform()
    valid_tile_dataset.transform = get_eval_patch_transform()
    test_tile_dataset.transform = get_eval_patch_transform()

m1_config = M1ModelConfig(
    feature_dim=M1_FEATURE_DIM,
    coord_dim=6,
    spatial_dim=SPATIAL_DIM,
    fusion_dim=FUSION_DIM,
    mil_hidden_dim=MIL_HIDDEN_DIM,
    n_outputs=N_OUTPUTS,
    dropout=DROPOUT,
    max_tiles=MAX_TILES_PER_SLIDE,
    feature_batch_size=FEATURE_BATCH_SIZE,
    freeze_feature_extractor=True,
    use_spatial_embedding=USE_SPATIAL_EMBEDDING,
    pooling_mode=POOLING_MODE,
)


def m1_loss_fn(logits: torch.Tensor, os_time_days: torch.Tensor, os_event: torch.Tensor) -> torch.Tensor:
    return cox_ph_loss(
        risk_scores=logits.float().reshape(-1),
        times=os_time_days.float().reshape(-1),
        events=os_event.float().reshape(-1),
    )

def initialize_m1_model(
    feature_extractor: nn.Module,
    feature_dim: int = M1_FEATURE_DIM,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[PathologySpatialMIL, torch.optim.Optimizer]:
    config = M1ModelConfig(
        feature_dim=feature_dim,
        coord_dim=6,
        spatial_dim=SPATIAL_DIM,
        fusion_dim=FUSION_DIM,
        mil_hidden_dim=MIL_HIDDEN_DIM,
        n_outputs=N_OUTPUTS,
        dropout=DROPOUT,
        max_tiles=MAX_TILES_PER_SLIDE,
        feature_batch_size=FEATURE_BATCH_SIZE,
        freeze_feature_extractor=True,
        use_spatial_embedding=USE_SPATIAL_EMBEDDING,
        pooling_mode=POOLING_MODE,
    )
    model = PathologySpatialMIL(feature_extractor=feature_extractor, config=config).to(device)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    return model, optimizer


model, optimizer = initialize_m1_model(feature_extractor, feature_dim=M1_FEATURE_DIM)

print("M1 model initialized.")
print("FEATURE_EXTRACTOR_NAME:", FEATURE_EXTRACTOR_NAME)
print("M1_FEATURE_DIM:", M1_FEATURE_DIM)
print("trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("MAX_TILES_PER_SLIDE:", MAX_TILES_PER_SLIDE)
print("FEATURE_BATCH_SIZE:", FEATURE_BATCH_SIZE)
print("USE_SPATIAL_EMBEDDING:", USE_SPATIAL_EMBEDDING)
print("POOLING_MODE:", POOLING_MODE)
print("N_OUTPUTS:", N_OUTPUTS, "patient-level risk score")

# train loop에서 사용할 loss 예시:
# outputs = model(tile_images, coords)
# loss = m1_loss_fn(outputs["logits"], labels)
# hazard_percent = outputs["hazard_percent"]
# dead_probability_percent = outputs["hazard_percent"]
