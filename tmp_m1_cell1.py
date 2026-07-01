# M1 TCGA-only 데이터 로드
# 입력 단위: 1 slide(case)의 tile image paths + tile 좌표 + slide 전체 크기 + multi-horizon survival label/mask
# 출력 label: dead by 6/12/18/24 months. Unknown(censored before horizon)은 mask=0으로 loss에서 제외합니다.

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_PATH = Path("../../data")
RESULT_PATH = Path("../../results")
PROJECT_DATA_PATH = DATA_PATH / "pancreatic_cancer_pathology"
DST_PATH = PROJECT_DATA_PATH / "dst"
IMAGE_PATH = DST_PATH / "Image"
CLINICAL_PATH = DST_PATH / "Clinical"

M1_OUTPUT_PATH = RESULT_PATH / "pancreatic_cancer_pathology" / "M1"
M1_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DATASET_NAMES = ["TCGA_PAAD", "CPTAC_PDAC"]
DATASET_NAME = "TCGA_CPTAC"
SEED = 42
PATCH_INPUT_SIZE = 512  # 1.0 MPP 1024 tile을 512 입력으로 줄이면 effective MPP는 약 2.0입니다.
PRELOAD_RESIZED_TILES = True  # dataset 구성 단계에서 512x512 tile을 CPU memory에 preload합니다.
PRELOAD_TILE_SIZE = PATCH_INPUT_SIZE
PATCH_MEAN = (0.485, 0.456, 0.406)
PATCH_STD = (0.229, 0.224, 0.225)
TEST_SIZE = 0.2
VALID_SIZE = 0.25  # train_valid 내부 비율. 전체 기준 0.8 * 0.25 = 0.2
REQUIRE_COMPLETE_24M_HORIZONS = True

MONTH_DAYS = 30.4375
HORIZON_MONTHS = [6, 12, 18, 24]
HORIZON_DAYS = np.array([m * MONTH_DAYS for m in HORIZON_MONTHS], dtype=np.float32)
HORIZON_NAMES = [f"dead_by_{m}m" for m in HORIZON_MONTHS]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_clinical_label(dataset: str, case_id: str) -> tuple[float | None, int | None]:
    clinical_json_path = CLINICAL_PATH / dataset / f"{case_id}_clinical.json"
    if not clinical_json_path.exists():
        return None, None
    with open(clinical_json_path, "r", encoding="utf-8") as f:
        clinical_json = json.load(f)
    clinical = clinical_json.get("clinical", {})
    return clinical.get("os_time_days"), clinical.get("os_event")


def make_horizon_label_mask(os_time_days: float, os_event: int) -> tuple[np.ndarray, np.ndarray]:
    """Create multi-horizon dead-by labels and known-label mask.

    label[h] = 1 if death occurred by horizon h, else 0.
    mask[h] = 1 if the label at horizon h is known.

    Censored before a horizon is unknown and excluded from BCE loss with mask=0.
    """
    y = np.zeros(len(HORIZON_DAYS), dtype=np.float32)
    mask = np.zeros(len(HORIZON_DAYS), dtype=np.float32)
    if pd.isna(os_time_days) or pd.isna(os_event):
        return y, mask

    os_time_days = float(os_time_days)
    os_event = int(os_event)

    for i, horizon in enumerate(HORIZON_DAYS):
        if os_event == 1 and os_time_days <= float(horizon):
            y[i] = 1.0
            mask[i] = 1.0
        elif os_time_days >= float(horizon):
            y[i] = 0.0
            mask[i] = 1.0
        else:
            # Censored before this horizon: unknown.
            y[i] = 0.0
            mask[i] = 0.0
    return y, mask


def get_patch_padding(image_size: int = PATCH_INPUT_SIZE) -> tuple[int, int, int, int]:
    # UNI2-h는 patch_size=14라 512 입력이 바로 들어갈 수 없습니다.
    # 512로 먼저 resize한 뒤 가장 가까운 patch-size multiple까지 symmetric padding합니다.
    patch_size = int(globals().get("FEATURE_EXTRACTOR_PATCH_SIZE", 16))
    target_size = int(np.ceil(image_size / patch_size) * patch_size)
    pad_total = max(0, target_size - image_size)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return (pad_left, pad_left, pad_right, pad_right)


def get_model_input_size(image_size: int = PATCH_INPUT_SIZE) -> int:
    patch_size = int(globals().get("FEATURE_EXTRACTOR_PATCH_SIZE", 16))
    return int(np.ceil(image_size / patch_size) * patch_size)


def get_train_patch_transform(image_size: int = PATCH_INPUT_SIZE):
    # Resize를 augmentation보다 먼저 수행해 1024 tile에서 바로 연산하지 않도록 합니다.
    # 이후 patch-size multiple로 padding합니다. UNI2-h는 512 -> 518 padding이 필요합니다.
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Pad(get_patch_padding(image_size), fill=255),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=PATCH_MEAN, std=PATCH_STD),
    ])


def get_eval_patch_transform(image_size: int = PATCH_INPUT_SIZE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.Pad(get_patch_padding(image_size), fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=PATCH_MEAN, std=PATCH_STD),
    ])


def get_train_cached_patch_transform(image_size: int = PRELOAD_TILE_SIZE):
    # dataset 구성 단계에서 이미 512x512로 resize된 image에 대해 padding/augmentation/normalization만 수행합니다.
    return transforms.Compose([
        transforms.Pad(get_patch_padding(image_size), fill=255),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=PATCH_MEAN, std=PATCH_STD),
    ])


def get_eval_cached_patch_transform(image_size: int = PRELOAD_TILE_SIZE):
    return transforms.Compose([
        transforms.Pad(get_patch_padding(image_size), fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=PATCH_MEAN, std=PATCH_STD),
    ])


def estimate_tile_cache_gb(n_tiles: int, image_size: int = PRELOAD_TILE_SIZE) -> float:
    return n_tiles * image_size * image_size * 3 / (1024 ** 3)


def preload_resized_tile_images(tile_index: pd.DataFrame, image_size: int = PRELOAD_TILE_SIZE) -> dict[str, np.ndarray]:
    unique_paths = sorted(tile_index["tile_path"].astype(str).unique())
    expected_gb = estimate_tile_cache_gb(len(unique_paths), image_size=image_size)
    print(f"Preloading resized tiles: {len(unique_paths):,} tiles, {image_size}x{image_size}, expected uint8 memory ~{expected_gb:.2f} GB")

    cache = {}
    resize_size = (image_size, image_size)
    for tile_path in tqdm(unique_paths, desc="Preload resized tile images", unit="tile"):
        with Image.open(tile_path) as image:
            image = image.convert("RGB")
            image = image.resize(resize_size, resample=Image.BILINEAR)
            cache[tile_path] = np.asarray(image, dtype=np.uint8).copy()
    return cache


def add_tile_coordinates(tile_df: pd.DataFrame) -> pd.DataFrame:
    tile_df = tile_df.copy()
    if {"x_level0", "y_level0"}.issubset(tile_df.columns):
        tile_df["x"] = tile_df["x_level0"].astype(int)
        tile_df["y"] = tile_df["y_level0"].astype(int)
    elif {"x_total_matrix", "y_total_matrix"}.issubset(tile_df.columns):
        tile_df["x"] = tile_df["x_total_matrix"].astype(int)
        tile_df["y"] = tile_df["y_total_matrix"].astype(int)
    else:
        raise ValueError("tile metadata must contain x/y coordinate columns.")
    return tile_df


def get_slide_dimensions(tile_df: pd.DataFrame) -> tuple[int, int]:
    source_size = tile_df["source_tile_size_px"].astype(float)
    width = int((tile_df["x"].astype(float) + source_size).max())
    height = int((tile_df["y"].astype(float) + source_size).max())
    return width, height


def load_case_tiles(dataset: str, metadata_path: Path) -> tuple[pd.DataFrame, dict]:
    case_id = metadata_path.parent.name
    case_dir = metadata_path.parent
    tile_df = pd.read_csv(metadata_path)

    tile_df = add_tile_coordinates(tile_df)
    slide_width, slide_height = get_slide_dimensions(tile_df)

    tile_df["tile_path"] = tile_df["tile_path"].astype(str)
    tile_df = tile_df[tile_df["tile_path"].map(lambda x: Path(x).exists())].copy()
    tile_df["slide_width"] = slide_width
    tile_df["slide_height"] = slide_height

    source_size = tile_df["source_tile_size_px"].astype(float)
    tile_df["x_norm"] = tile_df["x"].astype(float) / slide_width
    tile_df["y_norm"] = tile_df["y"].astype(float) / slide_height
    tile_df["x_center_norm"] = (tile_df["x"].astype(float) + source_size / 2) / slide_width
    tile_df["y_center_norm"] = (tile_df["y"].astype(float) + source_size / 2) / slide_height
    tile_df["w_norm"] = source_size / slide_width
    tile_df["h_norm"] = source_size / slide_height

    os_time, os_event = load_clinical_label(dataset, case_id)
    if os_time is None or os_event is None:
        os_time = tile_df["OS_time"].iloc[0] if "OS_time" in tile_df.columns else np.nan
        os_event = tile_df["OS_event"].iloc[0] if "OS_event" in tile_df.columns else np.nan

    y, mask = make_horizon_label_mask(os_time, os_event)
    case_record = {
        "dataset": dataset,
        "slide_uid": f"{dataset}::{case_id}",
        "case_id": case_id,
        "case_dir": case_dir.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "n_tiles": int(len(tile_df)),
        "slide_width": slide_width,
        "slide_height": slide_height,
        "os_time_days": float(os_time) if pd.notna(os_time) else np.nan,
        "os_event": int(os_event) if pd.notna(os_event) else np.nan,
        "known_horizon_count": int(mask.sum()),
    }
    for i, name in enumerate(HORIZON_NAMES):
        case_record[name] = float(y[i])
        case_record[f"mask_{name}"] = float(mask[i])
    tile_df["dataset"] = dataset
    tile_df["case_id"] = case_id
    tile_df["slide_uid"] = case_record["slide_uid"]
    return tile_df, case_record


case_records = []
tile_index_list = []
for dataset in DATASET_NAMES:
    metadata_paths = sorted((IMAGE_PATH / dataset).glob("*/tile_metadata.csv"))
    for metadata_path in tqdm(metadata_paths, desc=f"Load {dataset} metadata"):
        tile_df, case_record = load_case_tiles(dataset, metadata_path)
        if case_record["n_tiles"] == 0:
            case_record["exclude_reason"] = "no_tiles"
            case_records.append(case_record)
            continue
        tile_index_list.append(tile_df)
        case_records.append(case_record)

all_slide_df = pd.DataFrame(case_records)
tile_index_df = pd.concat(tile_index_list, ignore_index=True)


complete_horizon_mask = all_slide_df[[f"mask_{name}" for name in HORIZON_NAMES]].eq(1).all(axis=1)
required_horizon_mask = complete_horizon_mask if REQUIRE_COMPLETE_24M_HORIZONS else all_slide_df["known_horizon_count"].gt(0)

eligible_mask = (
    all_slide_df["os_time_days"].notna()
    & all_slide_df["os_event"].notna()
    & required_horizon_mask
    & all_slide_df["n_tiles"].gt(0)
)
excluded_df = all_slide_df[~eligible_mask].copy()
if not excluded_df.empty:
    excluded_df["exclude_reason"] = np.select(
        [
            excluded_df["n_tiles"].le(0),
            excluded_df["os_time_days"].isna(),
            excluded_df["os_event"].isna(),
            ~required_horizon_mask.loc[excluded_df.index],
        ],
        ["no_tiles", "missing_os_time", "missing_os_event", "incomplete_required_horizon"],
        default="unknown",
    )

slide_df = all_slide_df[eligible_mask].copy()
slide_df["os_event"] = slide_df["os_event"].astype(int)
tile_index_df = tile_index_df[tile_index_df["slide_uid"].isin(slide_df["slide_uid"])].copy()

slide_df.to_csv(M1_OUTPUT_PATH / "m1_tcga_cptac_horizon_slide_manifest.csv", index=False)
tile_index_df.to_csv(M1_OUTPUT_PATH / "m1_tcga_cptac_horizon_tile_index.csv", index=False)
excluded_df.to_csv(M1_OUTPUT_PATH / "m1_tcga_cptac_horizon_excluded_cases.csv", index=False)

print("all_slide_df:", all_slide_df.shape)
print("complete 24m horizon cases:", int(complete_horizon_mask.sum()))
print("slide_df:", slide_df.shape)
print("excluded_df:", excluded_df.shape)
print("tile_index_df:", tile_index_df.shape)

if PRELOAD_RESIZED_TILES:
    if "TILE_IMAGE_CACHE" not in globals() or not TILE_IMAGE_CACHE:
        TILE_IMAGE_CACHE = preload_resized_tile_images(tile_index_df, image_size=PRELOAD_TILE_SIZE)
    else:
        print(f"Using existing TILE_IMAGE_CACHE: {len(TILE_IMAGE_CACHE):,} tiles")
else:
    TILE_IMAGE_CACHE = {}

print("PRELOAD_RESIZED_TILES:", PRELOAD_RESIZED_TILES)
print("PRELOAD_TILE_SIZE:", PRELOAD_TILE_SIZE)
print("cached tiles:", len(TILE_IMAGE_CACHE))

horizon_summary = []
for name in HORIZON_NAMES:
    mask_col = f"mask_{name}"
    known = slide_df[mask_col].eq(1)
    horizon_summary.append({
        "horizon": name,
        "known_n": int(known.sum()),
        "dead_n": int(slide_df.loc[known, name].sum()),
        "alive_n": int(known.sum() - slide_df.loc[known, name].sum()),
        "unknown_n": int((~known).sum()),
        "dead_rate_known": float(slide_df.loc[known, name].mean()) if known.any() else np.nan,
    })
horizon_summary_df = pd.DataFrame(horizon_summary)
display(horizon_summary_df)
display(slide_df[["case_id", "n_tiles", "os_time_days", "os_event", "known_horizon_count"] + HORIZON_NAMES + [f"mask_{n}" for n in HORIZON_NAMES]].head())


class M1SlideDataset(Dataset):
    """TCGA case-level pathology dataset for multi-horizon MIL training."""

    def __init__(self, slide_manifest: pd.DataFrame, tile_index: pd.DataFrame):
        self.slide_manifest = slide_manifest.reset_index(drop=True).copy()
        self.tile_groups = {
            slide_uid: group.sort_values(["y", "x"]).reset_index(drop=True)
            for slide_uid, group in tile_index.groupby("slide_uid")
        }

    def __len__(self):
        return len(self.slide_manifest)

    def __getitem__(self, idx):
        row = self.slide_manifest.iloc[idx]
        tiles = self.tile_groups[row["slide_uid"]]

        coords = tiles[["x_norm", "y_norm", "x_center_norm", "y_center_norm", "w_norm", "h_norm"]].to_numpy(np.float32)
        label = row[HORIZON_NAMES].to_numpy(np.float32)
        slide_size = np.array([row["slide_width"], row["slide_height"]], dtype=np.float32)

        return {
            "dataset": row["dataset"],
            "case_id": row["case_id"],
            "slide_uid": row["slide_uid"],
            "tile_paths": tiles["tile_path"].tolist(),
            "coords": torch.from_numpy(coords),
            "slide_size": torch.from_numpy(slide_size),
            "horizon_months": torch.tensor(HORIZON_MONTHS, dtype=torch.float32),
            "label": torch.from_numpy(label),
            "os_time_days": torch.tensor(float(row["os_time_days"]), dtype=torch.float32),
            "os_event": torch.tensor(int(row["os_event"]), dtype=torch.long),
        }


class M1TileDataset(Dataset):
    """Tile-level dataset for UNI/UNI v2 on-the-fly feature extraction with augmentation."""

    def __init__(self, tile_index: pd.DataFrame, transform=None, tile_cache: dict[str, np.ndarray] | None = None):
        self.tile_index = tile_index.reset_index(drop=True).copy()
        self.transform = transform
        self.tile_cache = tile_cache or {}

    def __len__(self):
        return len(self.tile_index)

    def __getitem__(self, idx):
        row = self.tile_index.iloc[idx]
        tile_path = row["tile_path"]
        if tile_path in self.tile_cache:
            image = Image.fromarray(self.tile_cache[tile_path])
        else:
            with Image.open(tile_path) as image_file:
                image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        coords = row[["x_norm", "y_norm", "x_center_norm", "y_center_norm", "w_norm", "h_norm"]].to_numpy(np.float32)
        return {
            "image": image,
            "coords": torch.from_numpy(coords),
            "case_id": row["case_id"],
            "slide_uid": row["slide_uid"],
            "tile_path": tile_path,
        }


# 24개월까지 모든 horizon label이 확인 가능한 TCGA case만 사용하고, 전체 기준 6:2:2로 split합니다.
slide_df["stratify_group"] = slide_df["dataset"].astype(str) + "_event" + slide_df["os_event"].astype(str)
stratify_for_test = slide_df["stratify_group"] if slide_df["stratify_group"].value_counts().min() >= 2 else slide_df["os_event"]
train_valid_df, test_df = train_test_split(
    slide_df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=stratify_for_test,
)
stratify_for_valid = train_valid_df["stratify_group"] if train_valid_df["stratify_group"].value_counts().min() >= 2 else train_valid_df["os_event"]
train_df, valid_df = train_test_split(
    train_valid_df,
    test_size=VALID_SIZE,
    random_state=SEED,
    stratify=stratify_for_valid,
)

train_case_ids = set(train_df["slide_uid"])
valid_case_ids = set(valid_df["slide_uid"])
test_case_ids = set(test_df["slide_uid"])
assert len(train_case_ids & valid_case_ids) == 0
assert len(train_case_ids & test_case_ids) == 0
assert len(valid_case_ids & test_case_ids) == 0

train_dataset = M1SlideDataset(train_df, tile_index_df)
valid_dataset = M1SlideDataset(valid_df, tile_index_df)
test_dataset = M1SlideDataset(test_df, tile_index_df)

tile_train_transform = get_train_cached_patch_transform() if TILE_IMAGE_CACHE else get_train_patch_transform()
tile_eval_transform = get_eval_cached_patch_transform() if TILE_IMAGE_CACHE else get_eval_patch_transform()

train_tile_dataset = M1TileDataset(
    tile_index_df[tile_index_df["slide_uid"].isin(train_case_ids)],
    transform=tile_train_transform,
    tile_cache=TILE_IMAGE_CACHE,
)
valid_tile_dataset = M1TileDataset(
    tile_index_df[tile_index_df["slide_uid"].isin(valid_case_ids)],
    transform=tile_eval_transform,
    tile_cache=TILE_IMAGE_CACHE,
)
test_tile_dataset = M1TileDataset(
    tile_index_df[tile_index_df["slide_uid"].isin(test_case_ids)],
    transform=tile_eval_transform,
    tile_cache=TILE_IMAGE_CACHE,
)

split_df = slide_df[["dataset", "slide_uid", "case_id", "os_time_days", "os_event", "known_horizon_count"] + HORIZON_NAMES + [f"mask_{n}" for n in HORIZON_NAMES]].copy()
split_df["split"] = "unused"
split_df.loc[split_df["slide_uid"].isin(train_case_ids), "split"] = "train"
split_df.loc[split_df["slide_uid"].isin(valid_case_ids), "split"] = "valid"
split_df.loc[split_df["slide_uid"].isin(test_case_ids), "split"] = "test"
split_df.to_csv(M1_OUTPUT_PATH / "m1_tcga_cptac_horizon_case_splits.csv", index=False)

print("slide splits:", len(train_dataset), len(valid_dataset), len(test_dataset))
print("tile splits:", len(train_tile_dataset), len(valid_tile_dataset), len(test_tile_dataset))
print("split x dataset")
display(pd.crosstab(split_df["split"], split_df["dataset"]))
print("split x os_event")
display(pd.crosstab(split_df["split"], split_df["os_event"]))
print("split x dataset x os_event")
display(pd.crosstab([split_df["split"], split_df["dataset"]], split_df["os_event"]))

split_horizon_summary = []
for split_name, part in split_df.groupby("split"):
    row = {"split": split_name, "n_cases": len(part)}
    for name in HORIZON_NAMES:
        known = part[f"mask_{name}"].eq(1)
        row[f"{name}_known"] = int(known.sum())
        row[f"{name}_dead"] = int(part.loc[known, name].sum())
    split_horizon_summary.append(row)
display(pd.DataFrame(split_horizon_summary))

sample = train_dataset[0]
print("sample case:", sample["case_id"])
print("n_tiles:", len(sample["tile_paths"]))
print("coords:", sample["coords"].shape, sample["coords"][:3])
print("slide_size:", sample["slide_size"])
print("label:", sample["label"])
print("model input size:", get_model_input_size(PATCH_INPUT_SIZE), "padding:", get_patch_padding(PATCH_INPUT_SIZE))
print("model output 해석: logits -> sigmoid(logits) * 100 = horizon별 사망위험 percent")
