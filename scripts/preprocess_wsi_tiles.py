from __future__ import annotations

import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import openslide
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixels import pixel_array as dicom_pixel_array
from tqdm.auto import tqdm


DATA_PATH = Path("../../data")
PROJECT_DATA_PATH = DATA_PATH / "pancreatic_cancer_pathology"
RAW_PATH = PROJECT_DATA_PATH / "raw"
DST_PATH = PROJECT_DATA_PATH / "dst"
IMAGE_DST_PATH = DST_PATH / "Image"
TCGA_RAW_PATH = RAW_PATH / "TCGA_PAAD"
CPTAC_RAW_PATH = RAW_PATH / "CPTAC_PDAC"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_name(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def tissue_fraction(rgb: np.ndarray) -> float:
    if rgb.size == 0:
        return 0.0
    rgb = rgb[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    tissue_mask = (saturation > 20) & (value < 245)
    return float(tissue_mask.mean())


def tissue_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb[..., :3].astype(np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] > 20) & (hsv[..., 2] < 245)


def save_tile(tile: np.ndarray, out_path: Path, image_format: str = "png", jpeg_quality: int = 90) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(tile.astype(np.uint8))
    if image_format.lower() in {"jpg", "jpeg"}:
        image.save(out_path, quality=jpeg_quality)
    else:
        image.save(out_path)


def get_openslide_mpp(slide: openslide.OpenSlide) -> float:
    props = slide.properties
    for key in [openslide.PROPERTY_NAME_MPP_X, "aperio.MPP", "openslide.mpp-x"]:
        value = props.get(key)
        if value is not None:
            return float(value)
    objective = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if objective is not None:
        objective = float(objective)
        if objective > 0:
            return 10.0 / objective
    raise ValueError("WSI mpp 정보를 찾지 못했습니다. slide.properties를 확인하세요.")


def get_dicom_mpp(ds: pydicom.Dataset) -> float:
    if hasattr(ds, "SharedFunctionalGroupsSequence"):
        shared = ds.SharedFunctionalGroupsSequence[0]
        if hasattr(shared, "PixelMeasuresSequence"):
            spacing = shared.PixelMeasuresSequence[0].PixelSpacing
            return float(spacing[0]) * 1000.0
    for key in ["PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"]:
        spacing = getattr(ds, key, None)
        if spacing is not None:
            return float(spacing[0]) * 1000.0
    raise ValueError("DICOM mpp 정보를 찾지 못했습니다.")


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[-1] in (3, 4):
        return frame[..., :3]
    if frame.ndim == 2:
        return np.repeat(frame[..., None], 3, axis=-1)
    raise ValueError(f"예상하지 못한 frame shape입니다: {frame.shape}")


def resize_to_target(tile: np.ndarray, tile_size: int) -> np.ndarray:
    return cv2.resize(tile[..., :3], (tile_size, tile_size), interpolation=cv2.INTER_AREA)


def build_tcga_tissue_coords(
    slide: openslide.OpenSlide,
    source_tile_size: int,
    prefilter_tissue_threshold: float,
    tissue_mask_downsample: int,
) -> tuple[list[tuple[int, int, float]], dict[str, object]]:
    width, height = slide.dimensions
    level = slide.get_best_level_for_downsample(tissue_mask_downsample)
    level_downsample = float(slide.level_downsamples[level])
    level_width, level_height = slide.level_dimensions[level]

    thumbnail = slide.read_region((0, 0), level, (level_width, level_height)).convert("RGB")
    mask = tissue_mask_from_rgb(np.asarray(thumbnail))

    coords = []
    n_candidate = 0
    for y in range(0, height - source_tile_size + 1, source_tile_size):
        for x in range(0, width - source_tile_size + 1, source_tile_size):
            n_candidate += 1
            x0 = max(0, int(x / level_downsample))
            y0 = max(0, int(y / level_downsample))
            x1 = min(level_width, int(math.ceil((x + source_tile_size) / level_downsample)))
            y1 = min(level_height, int(math.ceil((y + source_tile_size) / level_downsample)))
            if x1 <= x0 or y1 <= y0:
                continue
            prefilter_fraction = float(mask[y0:y1, x0:x1].mean())
            if prefilter_fraction >= prefilter_tissue_threshold:
                coords.append((x, y, prefilter_fraction))

    info = {
        "slide_width": width,
        "slide_height": height,
        "mask_level": level,
        "mask_level_downsample": level_downsample,
        "mask_width": level_width,
        "mask_height": level_height,
        "n_candidate_tiles": n_candidate,
        "n_prefiltered_tiles": len(coords),
    }
    return coords, info


def tile_tcga_case(
    row_dict: dict,
    target_mpp: float = 1.0,
    tile_size: int = 1024,
    tissue_threshold: float = 0.15,
    prefilter_tissue_threshold: float = 0.01,
    tissue_mask_downsample: int = 64,
    skip_existing: bool = True,
    overwrite_metadata: bool = False,
    image_format: str = "png",
    jpeg_quality: int = 90,
    show_tile_progress: bool = False,
) -> dict[str, object]:
    dataset_name = "TCGA_PAAD"
    image_dst = IMAGE_DST_PATH / dataset_name
    case_id = safe_name(row_dict["patient_id"])
    wsi_path = Path(row_dict["wsi_path"])
    case_dir = image_dst / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = case_dir / "tile_metadata.csv"

    if skip_existing and metadata_path.exists() and not overwrite_metadata:
        return {
            "dataset": dataset_name,
            "case_id": case_id,
            "status": "skipped_existing",
            "metadata_path": metadata_path.as_posix(),
            "case_dir": case_dir.as_posix(),
        }

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        native_mpp = get_openslide_mpp(slide)
        source_tile_size = int(round(tile_size * target_mpp / native_mpp))
        target_downsample = target_mpp / native_mpp
        read_level = min(
            range(slide.level_count),
            key=lambda level_idx: abs(float(slide.level_downsamples[level_idx]) - target_downsample),
        )
        read_level_downsample = float(slide.level_downsamples[read_level])
        read_level_mpp = native_mpp * read_level_downsample
        read_level_tile_size = max(
            1,
            int(round(tile_size * target_mpp / read_level_mpp)),
        )
        coords, mask_info = build_tcga_tissue_coords(
            slide,
            source_tile_size=source_tile_size,
            prefilter_tissue_threshold=prefilter_tissue_threshold,
            tissue_mask_downsample=tissue_mask_downsample,
        )

        records = []
        saved = 0
        suffix = "jpg" if image_format.lower() in {"jpg", "jpeg"} else "png"
        tile_iter = tqdm(
            list(enumerate(coords)),
            desc=f"TCGA {case_id} tiles",
            leave=False,
            disable=not show_tile_progress,
        )
        for tile_idx, (x, y, prefilter_fraction) in tile_iter:
            out_path = case_dir / f"{case_id}_x{x}_y{y}_mpp{target_mpp:.1f}_{tile_idx:06d}.{suffix}"
            if skip_existing and out_path.exists():
                continue
            region = slide.read_region((x, y), read_level, (read_level_tile_size, read_level_tile_size)).convert("RGB")
            tile = cv2.resize(
                np.asarray(region),
                (tile_size, tile_size),
                interpolation=cv2.INTER_AREA if read_level_tile_size > tile_size else cv2.INTER_CUBIC,
            )
            frac = tissue_fraction(tile)
            if frac < tissue_threshold:
                continue
            save_tile(tile, out_path, image_format=image_format, jpeg_quality=jpeg_quality)
            saved += 1
            if show_tile_progress:
                tile_iter.set_postfix(saved=saved)
            records.append(
                {
                    "dataset": dataset_name,
                    "case_id": case_id,
                    "tile_path": out_path.as_posix(),
                    "wsi_path": wsi_path.as_posix(),
                    "x_level0": x,
                    "y_level0": y,
                    "source_tile_size_px": source_tile_size,
                    "target_tile_size_px": tile_size,
                    "native_mpp": native_mpp,
                    "read_level": read_level,
                    "read_level_downsample": read_level_downsample,
                    "read_level_mpp": read_level_mpp,
                    "read_level_tile_size_px": read_level_tile_size,
                    "target_mpp": target_mpp,
                    "prefilter_tissue_fraction": prefilter_fraction,
                    "tissue_fraction": frac,
                    "OS_time": row_dict.get("OS_time"),
                    "OS_event": row_dict.get("OS_event"),
                }
            )

        pd.DataFrame(records).to_csv(metadata_path, index=False)
        return {
            "dataset": dataset_name,
            "case_id": case_id,
            "status": "done",
            "wsi_path": wsi_path.as_posix(),
            "native_mpp": native_mpp,
            "read_level": read_level,
            "read_level_downsample": read_level_downsample,
            "read_level_mpp": read_level_mpp,
            "read_level_tile_size_px": read_level_tile_size,
            "source_tile_size_px": source_tile_size,
            "n_candidate_tiles": mask_info["n_candidate_tiles"],
            "n_prefiltered_tiles": mask_info["n_prefiltered_tiles"],
            "n_saved_tiles": saved,
            "case_dir": case_dir.as_posix(),
            "metadata_path": metadata_path.as_posix(),
        } | mask_info
    finally:
        slide.close()


def select_cptac_volume_dicom(series_dir: Path, target_mpp: float, prefer_finer_or_equal: bool = True) -> tuple[Path, pd.DataFrame]:
    records = []
    for path in sorted(series_dir.glob("*.dcm"), key=lambda p: p.stat().st_size):
        ds_meta = pydicom.dcmread(path, stop_before_pixels=True)
        image_type = " | ".join(map(str, getattr(ds_meta, "ImageType", []))).upper()
        if "VOLUME" not in image_type:
            continue
        mpp = get_dicom_mpp(ds_meta)
        records.append(
            {
                "path": path,
                "file_name": path.name,
                "image_type": image_type,
                "mpp": mpp,
                "rows": int(ds_meta.Rows),
                "columns": int(ds_meta.Columns),
                "number_of_frames": int(getattr(ds_meta, "NumberOfFrames", 1)),
                "total_rows": int(getattr(ds_meta, "TotalPixelMatrixRows", ds_meta.Rows)),
                "total_cols": int(getattr(ds_meta, "TotalPixelMatrixColumns", ds_meta.Columns)),
                "size_mb": path.stat().st_size / (1024 ** 2),
            }
        )
    meta_df = pd.DataFrame(records)
    if meta_df.empty:
        raise ValueError(f"VOLUME DICOM을 찾지 못했습니다: {series_dir}")
    if prefer_finer_or_equal:
        candidates = meta_df[meta_df["mpp"] <= target_mpp].copy()
        if candidates.empty:
            candidates = meta_df.copy()
        selected = candidates.sort_values("mpp", ascending=False).iloc[0]
    else:
        selected = meta_df.iloc[(meta_df["mpp"] - target_mpp).abs().argsort()].iloc[0]
    return Path(selected["path"]), meta_df


def read_dicom_frame(path: Path, frame_idx: int) -> np.ndarray:
    return normalize_frame(dicom_pixel_array(path, index=int(frame_idx)))


def get_dicom_region_partial(
    volume_path: Path,
    x: int,
    y: int,
    source_size: int,
    total_cols: int,
    total_rows: int,
    tile_cols: int,
    tile_rows: int,
) -> np.ndarray:
    grid_cols = math.ceil(total_cols / tile_cols)
    grid_rows = math.ceil(total_rows / tile_rows)
    region = np.full((source_size, source_size, 3), fill_value=255, dtype=np.uint8)

    start_col = x // tile_cols
    end_col = min(grid_cols - 1, (x + source_size - 1) // tile_cols)
    start_row = y // tile_rows
    end_row = min(grid_rows - 1, (y + source_size - 1) // tile_rows)

    for row_idx in range(start_row, end_row + 1):
        for col_idx in range(start_col, end_col + 1):
            frame_idx = row_idx * grid_cols + col_idx
            frame = read_dicom_frame(volume_path, frame_idx)
            frame_x0 = col_idx * tile_cols
            frame_y0 = row_idx * tile_rows
            overlap_x0 = max(x, frame_x0)
            overlap_y0 = max(y, frame_y0)
            overlap_x1 = min(x + source_size, frame_x0 + tile_cols, total_cols)
            overlap_y1 = min(y + source_size, frame_y0 + tile_rows, total_rows)
            if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
                continue
            src_x0 = overlap_x0 - frame_x0
            src_y0 = overlap_y0 - frame_y0
            dst_x0 = overlap_x0 - x
            dst_y0 = overlap_y0 - y
            width = overlap_x1 - overlap_x0
            height = overlap_y1 - overlap_y0
            region[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] = frame[
                src_y0 : src_y0 + height, src_x0 : src_x0 + width
            ]
    return region


def build_cptac_prefilter_coords(
    source_meta: pd.Series,
    mask_meta: pd.Series,
    source_tile_size: int,
    prefilter_tissue_threshold: float,
    show_progress: bool = False,
    case_id: str = "",
) -> tuple[list[tuple[int, int, float]], dict[str, object]]:
    source_total_cols = int(source_meta["total_cols"])
    source_total_rows = int(source_meta["total_rows"])
    source_mpp = float(source_meta["mpp"])
    mask_mpp = float(mask_meta["mpp"])
    scale = source_mpp / mask_mpp

    coords = []
    n_candidate = 0
    y_values = list(range(0, source_total_rows - source_tile_size + 1, source_tile_size))
    y_iter = tqdm(
        y_values,
        desc=f"CPTAC {case_id} prefilter",
        leave=False,
        disable=not show_progress,
    )
    for y in y_iter:
        for x in range(0, source_total_cols - source_tile_size + 1, source_tile_size):
            n_candidate += 1
            mask_x = int(round(x * scale))
            mask_y = int(round(y * scale))
            mask_size = max(1, int(round(source_tile_size * scale)))
            mask_region = get_dicom_region_partial(
                Path(mask_meta["path"]),
                mask_x,
                mask_y,
                mask_size,
                int(mask_meta["total_cols"]),
                int(mask_meta["total_rows"]),
                int(mask_meta["columns"]),
                int(mask_meta["rows"]),
            )
            prefilter_fraction = tissue_fraction(mask_region)
            if prefilter_fraction >= prefilter_tissue_threshold:
                coords.append((x, y, prefilter_fraction))
        if show_progress:
            y_iter.set_postfix(selected=len(coords), candidate=n_candidate)

    return coords, {
        "n_candidate_tiles": n_candidate,
        "n_prefiltered_tiles": len(coords),
        "prefilter_mask_mpp": mask_mpp,
        "prefilter_mask_path": Path(mask_meta["path"]).as_posix(),
    }


def tile_cptac_case(
    row_dict: dict,
    cptac_wsi_records: list[dict],
    target_mpp: float = 1.0,
    tile_size: int = 1024,
    tissue_threshold: float = 0.15,
    prefilter_tissue_threshold: float = 0.01,
    skip_existing: bool = True,
    overwrite_metadata: bool = False,
    image_format: str = "png",
    jpeg_quality: int = 90,
    show_tile_progress: bool = False,
) -> dict[str, object]:
    dataset_name = "CPTAC_PDAC"
    image_dst = IMAGE_DST_PATH / dataset_name
    case_id = safe_name(row_dict["case_id"])
    case_dir = image_dst / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = case_dir / "tile_metadata.csv"

    if skip_existing and metadata_path.exists() and not overwrite_metadata:
        return {
            "dataset": dataset_name,
            "case_id": case_id,
            "status": "skipped_existing",
            "metadata_path": metadata_path.as_posix(),
            "case_dir": case_dir.as_posix(),
        }

    cptac_wsi_df = pd.DataFrame(cptac_wsi_records)
    case_series = cptac_wsi_df[cptac_wsi_df["case_id"].eq(row_dict["case_id"])].copy()
    tumor_series = case_series[
        case_series["SeriesDescription"].astype(str).str.contains("tumor", case=False, na=False)
    ]
    if tumor_series.empty:
        return {"dataset": dataset_name, "case_id": case_id, "status": "no_tumor_series"}

    selected_series = tumor_series.sort_values("series_size_MB", ascending=False).iloc[0]
    series_dirs = list((CPTAC_RAW_PATH / "WSI_DICOM").rglob(f"SM_{selected_series['SeriesInstanceUID']}"))
    if len(series_dirs) == 0:
        return {"dataset": dataset_name, "case_id": case_id, "status": "missing_series_dir"}
    series_dir = series_dirs[0]

    source_volume_path, volume_meta_df = select_cptac_volume_dicom(series_dir, target_mpp, prefer_finer_or_equal=True)
    volume_meta_df.to_csv(case_dir / "dicom_volume_candidates.csv", index=False)
    source_meta = volume_meta_df[volume_meta_df["path"].eq(source_volume_path)].iloc[0]
    mask_meta = volume_meta_df.iloc[(volume_meta_df["mpp"] - max(target_mpp * 2, target_mpp)).abs().argsort()].iloc[0]

    native_mpp = float(source_meta["mpp"])
    source_tile_size = int(round(tile_size * target_mpp / native_mpp))
    coords, prefilter_info = build_cptac_prefilter_coords(
        source_meta=source_meta,
        mask_meta=mask_meta,
        source_tile_size=source_tile_size,
        prefilter_tissue_threshold=prefilter_tissue_threshold,
        show_progress=show_tile_progress,
        case_id=case_id,
    )

    records = []
    saved = 0
    suffix = "jpg" if image_format.lower() in {"jpg", "jpeg"} else "png"
    tile_iter = tqdm(
        list(enumerate(coords)),
        desc=f"CPTAC {case_id} tiles",
        leave=False,
        disable=not show_tile_progress,
    )
    for tile_idx, (x, y, prefilter_fraction) in tile_iter:
        out_path = case_dir / f"{case_id}_x{x}_y{y}_mpp{target_mpp:.1f}_{tile_idx:06d}.{suffix}"
        if skip_existing and out_path.exists():
            continue
        source_region = get_dicom_region_partial(
            source_volume_path,
            x,
            y,
            source_tile_size,
            int(source_meta["total_cols"]),
            int(source_meta["total_rows"]),
            int(source_meta["columns"]),
            int(source_meta["rows"]),
        )
        tile = resize_to_target(source_region, tile_size=tile_size)
        frac = tissue_fraction(tile)
        if frac < tissue_threshold:
            continue
        save_tile(tile, out_path, image_format=image_format, jpeg_quality=jpeg_quality)
        saved += 1
        if show_tile_progress:
            tile_iter.set_postfix(saved=saved)
        records.append(
            {
                "dataset": dataset_name,
                "case_id": case_id,
                "tile_path": out_path.as_posix(),
                "volume_dicom_path": source_volume_path.as_posix(),
                "SeriesInstanceUID": selected_series["SeriesInstanceUID"],
                "SeriesDescription": selected_series["SeriesDescription"],
                "x_total_matrix": x,
                "y_total_matrix": y,
                "source_tile_size_px": source_tile_size,
                "target_tile_size_px": tile_size,
                "native_mpp": native_mpp,
                "target_mpp": target_mpp,
                "prefilter_tissue_fraction": prefilter_fraction,
                "tissue_fraction": frac,
                "OS_time": row_dict.get("OS_time"),
                "OS_event": row_dict.get("OS_event"),
            }
        )

    pd.DataFrame(records).to_csv(metadata_path, index=False)
    return {
        "dataset": dataset_name,
        "case_id": case_id,
        "status": "done",
        "SeriesInstanceUID": selected_series["SeriesInstanceUID"],
        "SeriesDescription": selected_series["SeriesDescription"],
        "volume_dicom_path": source_volume_path.as_posix(),
        "native_mpp": native_mpp,
        "source_tile_size_px": source_tile_size,
        "n_saved_tiles": saved,
        "case_dir": case_dir.as_posix(),
        "metadata_path": metadata_path.as_posix(),
    } | prefilter_info


def run_parallel_jobs(jobs: list[tuple], max_workers: int, desc: str) -> list[dict[str, object]]:
    if max_workers <= 1:
        results = []
        for func, args, kwargs in tqdm(jobs, desc=desc, unit="case"):
            results.append(func(*args, **kwargs))
        return results

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args, **kwargs) for func, args, kwargs in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="case"):
            results.append(future.result())
    return results


def run_tcga_tiling(
    cohort_path: Path,
    max_cases: Optional[int] = None,
    max_workers: int = 4,
    **kwargs,
) -> pd.DataFrame:
    tcga_df = pd.read_csv(cohort_path)
    tcga_df = tcga_df[tcga_df["wsi_exists"].astype(bool)].copy()
    if max_cases is not None:
        tcga_df = tcga_df.head(max_cases).copy()
    kwargs = dict(kwargs)
    kwargs["show_tile_progress"] = bool(kwargs.get("show_tile_progress", max_workers <= 1))

    jobs = [(tile_tcga_case, (row.to_dict(),), kwargs) for _, row in tcga_df.iterrows()]
    results = run_parallel_jobs(jobs, max_workers=max_workers, desc="TCGA cases")
    summary_df = pd.DataFrame(results)
    out_path = IMAGE_DST_PATH / "TCGA_PAAD" / "tile_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    return summary_df


def run_cptac_tiling(
    cohort_path: Path,
    wsi_series_path: Path,
    max_cases: Optional[int] = None,
    max_workers: int = 2,
    **kwargs,
) -> pd.DataFrame:
    cptac_df = pd.read_csv(cohort_path)
    cptac_df = cptac_df[cptac_df["has_wsi_series"].astype(bool)].copy()
    if max_cases is not None:
        cptac_df = cptac_df.head(max_cases).copy()
    cptac_wsi_records = pd.read_csv(wsi_series_path).to_dict("records")
    kwargs = dict(kwargs)
    kwargs["show_tile_progress"] = bool(kwargs.get("show_tile_progress", max_workers <= 1))

    jobs = [
        (tile_cptac_case, (row.to_dict(), cptac_wsi_records), kwargs)
        for _, row in cptac_df.iterrows()
    ]
    results = run_parallel_jobs(jobs, max_workers=max_workers, desc="CPTAC cases")
    summary_df = pd.DataFrame(results)
    out_path = IMAGE_DST_PATH / "CPTAC_PDAC" / "tile_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    return summary_df
