from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.model_selection import train_test_split


DATA_PATH = Path("../../data")
RESULT_PATH = Path("../../results")
PROJECT_DATA_PATH = DATA_PATH / "pancreatic_cancer_pathology"
RNASEQ_DST_PATH = PROJECT_DATA_PATH / "dst" / "RNAseq"
CLINICAL_DST_PATH = PROJECT_DATA_PATH / "dst" / "Clinical"
SELECTED_RNASEQ_DST_PATH = PROJECT_DATA_PATH / "dst" / "RNAseq_selected"
FEATURE_SELECTION_RESULT_PATH = RESULT_PATH / "pancreatic_cancer_pathology" / "data_preprocessing" / "rnaseq_feature_selection"

DATASETS = ("TCGA_PAAD", "CPTAC_PDAC")
ZSCORE_MATRIX_NAME = "matrix_common_genes_zscore.csv"
DEFAULT_SPLIT_CANDIDATES = (
    RESULT_PATH / "pancreatic_cancer_pathology" / "M1" / "m1_tcga_cptac_horizon_case_splits.csv",
    Path("outputs/M1/m1_tcga_cptac_horizon_case_splits.csv"),
)

PDAC_LITERATURE_GENE_SETS = {
    "core_driver_tumor_suppressor": [
        "KRAS",
        "TP53",
        "CDKN2A",
        "CDKN2B",
        "SMAD4",
        "ARID1A",
        "KDM6A",
        "RNF43",
        "GNAS",
        "TGFBR2",
        "STK11",
        "SMARCA4",
        "PIK3CA",
        "PTEN",
        "BRAF",
        "MYC",
    ],
    "dna_damage_repair_therapy": [
        "BRCA1",
        "BRCA2",
        "PALB2",
        "ATM",
        "ATR",
        "CHEK1",
        "CHEK2",
        "RAD51",
        "MLH1",
        "MSH2",
        "MSH6",
        "PMS2",
        "ERCC1",
    ],
    "classical_pancreatic_progenitor": [
        "GATA6",
        "HNF1A",
        "HNF4A",
        "HNF4G",
        "FOXA2",
        "FOXA3",
        "PDX1",
        "MNX1",
        "ONECUT1",
        "ONECUT2",
        "KRT19",
        "EPCAM",
        "CDH1",
        "MUC1",
        "MUC5AC",
        "CEACAM5",
        "CEACAM6",
        "CLDN4",
        "CLDN18",
        "TFF1",
        "TFF2",
        "AGR2",
    ],
    "basal_squamous_mesenchymal": [
        "KRT5",
        "KRT6A",
        "KRT6B",
        "KRT14",
        "KRT17",
        "KRT81",
        "TP63",
        "KLF5",
        "S100A2",
        "S100A4",
        "SERPINB3",
        "SERPINB4",
        "VIM",
        "CDH2",
        "ZEB1",
        "ZEB2",
        "SNAI1",
        "SNAI2",
        "TWIST1",
        "ITGA6",
        "LAMC2",
    ],
    "stroma_ecm_invasion": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "COL5A2",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "FN1",
        "SPARC",
        "POSTN",
        "THBS1",
        "ACTA2",
        "TAGLN",
        "FAP",
        "ITGA2",
        "ITGA3",
        "ITGB1",
        "ITGB4",
        "MMP2",
        "MMP7",
        "MMP9",
        "MMP11",
        "MMP14",
        "PLAU",
        "PLAUR",
        "LOX",
        "LUM",
        "DCN",
        "BGN",
        "MET",
    ],
    "immune_inflammation_tgf_beta": [
        "CD274",
        "PDCD1",
        "CTLA4",
        "CD8A",
        "CD8B",
        "CD3D",
        "CD3E",
        "FOXP3",
        "CD68",
        "CD163",
        "LYZ",
        "CXCL12",
        "CXCR4",
        "CXCL8",
        "IL6",
        "IL6R",
        "STAT3",
        "TGFB1",
        "TGFB2",
        "TGFBR1",
        "TGFBR2",
        "CCL2",
        "CCR2",
        "CSF1",
        "CSF1R",
    ],
    "proliferation_cell_cycle_apoptosis": [
        "MKI67",
        "TOP2A",
        "CCNB1",
        "CCND1",
        "CCNE1",
        "CDK1",
        "CDK2",
        "BIRC5",
        "AURKA",
        "AURKB",
        "PLK1",
        "MCM2",
        "MCM4",
        "MCM6",
        "PCNA",
        "BCL2",
        "BAX",
        "CASP3",
    ],
    "hypoxia_metabolism_acinar_program": [
        "HIF1A",
        "VEGFA",
        "CA9",
        "SLC2A1",
        "LDHA",
        "HK2",
        "ENO1",
        "ALDOA",
        "PNLIP",
        "CPA1",
        "CPA2",
        "CPB1",
        "CTRB1",
        "CTRB2",
        "CLPS",
        "PRSS1",
        "REG1A",
        "REG1B",
    ],
}


def load_rnaseq_zscore_matrices() -> dict[str, pd.DataFrame]:
    matrices: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        matrix_path = RNASEQ_DST_PATH / dataset / ZSCORE_MATRIX_NAME
        if not matrix_path.exists():
            raise FileNotFoundError(f"RNA-seq matrix not found: {matrix_path}")
        matrix = pd.read_csv(matrix_path, index_col=0)
        matrix.index = matrix.index.astype(str)
        matrix.columns = matrix.columns.astype(str).str.upper()
        matrices[dataset] = matrix
    return matrices


def load_clinical_records() -> pd.DataFrame:
    records = []
    for dataset in DATASETS:
        for json_path in sorted((CLINICAL_DST_PATH / dataset).glob("*_clinical.json")):
            payload = json.loads(json_path.read_text())
            clinical = payload.get("clinical", {})
            case_id = str(payload.get("case_id") or json_path.name.replace("_clinical.json", ""))
            records.append(
                {
                    "dataset": dataset,
                    "case_id": case_id,
                    "slide_uid": f"{dataset}::{case_id}",
                    "os_time_days": clinical.get("os_time_days"),
                    "os_event": clinical.get("os_event"),
                    "has_rnaseq": clinical.get("has_rnaseq"),
                    "has_wsi": clinical.get("has_wsi"),
                }
            )
    df = pd.DataFrame(records)
    df["os_time_days"] = pd.to_numeric(df["os_time_days"], errors="coerce")
    df["os_event"] = pd.to_numeric(df["os_event"], errors="coerce")
    return df


def build_reproducible_split(seed: int = 42) -> pd.DataFrame:
    clinical_df = load_clinical_records()
    clinical_df = clinical_df[
        clinical_df["has_rnaseq"].eq(True)
        & clinical_df["os_time_days"].notna()
        & clinical_df["os_event"].isin([0, 1])
        & clinical_df["os_time_days"].gt(0)
    ].copy()
    clinical_df["strata"] = clinical_df["dataset"].astype(str) + "_event" + clinical_df["os_event"].astype(int).astype(str)

    train_df, temp_df = train_test_split(
        clinical_df,
        test_size=0.4,
        random_state=seed,
        stratify=clinical_df["strata"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["strata"],
    )
    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            valid_df.assign(split="valid"),
            test_df.assign(split="test"),
        ],
        axis=0,
    ).drop(columns="strata")
    return split_df.sort_values(["dataset", "case_id"]).reset_index(drop=True)


def load_or_build_split(split_path: Path | None = None, seed: int = 42) -> pd.DataFrame:
    candidate_paths = [split_path] if split_path is not None else list(DEFAULT_SPLIT_CANDIDATES)
    for path in candidate_paths:
        if path is not None and path.exists():
            split_df = pd.read_csv(path)
            break
    else:
        split_df = build_reproducible_split(seed=seed)

    required = {"dataset", "case_id", "os_time_days", "os_event", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split table is missing required columns: {sorted(missing)}")

    split_df = split_df.copy()
    split_df["dataset"] = split_df["dataset"].astype(str)
    split_df["case_id"] = split_df["case_id"].astype(str)
    split_df["slide_uid"] = split_df.get("slide_uid", split_df["dataset"] + "::" + split_df["case_id"])
    split_df["os_time_days"] = pd.to_numeric(split_df["os_time_days"], errors="coerce")
    split_df["os_event"] = pd.to_numeric(split_df["os_event"], errors="coerce")
    split_df = split_df[
        split_df["os_time_days"].notna()
        & split_df["os_event"].isin([0, 1])
        & split_df["os_time_days"].gt(0)
    ].copy()
    return split_df.reset_index(drop=True)


def cox_score_test_matrix(x: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-time)
    x = x[order].astype(np.float64, copy=False)
    event = event[order].astype(bool, copy=False)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    risk_count = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    risk_sum = np.cumsum(x, axis=0)
    risk_mean = risk_sum / risk_count[:, None]

    risk_sq_sum = np.cumsum(x * x, axis=0)
    risk_var = risk_sq_sum / risk_count[:, None] - risk_mean * risk_mean
    risk_var = np.clip(risk_var, 1e-12, None)

    event_x = x[event]
    event_mean = risk_mean[event]
    event_var = risk_var[event]
    u = (event_x - event_mean).sum(axis=0)
    info = event_var.sum(axis=0)
    z = u / np.sqrt(np.clip(info, 1e-12, None))
    chi2_stat = z * z
    p_value = chi2.sf(chi2_stat, df=1)
    return z, chi2_stat, p_value


def rank_genes_by_train_survival_cox(
    matrices: dict[str, pd.DataFrame],
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    common_genes = sorted(set.intersection(*(set(df.columns) for df in matrices.values())))
    if not common_genes:
        raise ValueError("No common RNA-seq genes found across datasets.")

    train_df = split_df[split_df["split"].eq("train")].copy()
    rows = pd.DataFrame({"gene_symbol": common_genes})
    z_columns = []

    for dataset, matrix in matrices.items():
        dataset_train = train_df[train_df["dataset"].eq(dataset)].copy()
        available_cases = [case_id for case_id in dataset_train["case_id"] if case_id in matrix.index]
        dataset_train = dataset_train.set_index("case_id").loc[available_cases].reset_index()
        x = matrix.loc[available_cases, common_genes].to_numpy(dtype=np.float64)
        time = dataset_train["os_time_days"].to_numpy(dtype=np.float64)
        event = dataset_train["os_event"].to_numpy(dtype=np.int64)

        prefix = dataset.lower()
        rows[f"{prefix}_train_n"] = len(dataset_train)
        rows[f"{prefix}_train_events"] = int(event.sum())
        if len(dataset_train) < 10 or event.sum() < 3:
            rows[f"{prefix}_cox_z"] = np.nan
            rows[f"{prefix}_cox_chi2"] = np.nan
            rows[f"{prefix}_cox_p"] = np.nan
            continue

        z, chi2_stat, p_value = cox_score_test_matrix(x=x, time=time, event=event)
        rows[f"{prefix}_cox_z"] = z
        rows[f"{prefix}_cox_chi2"] = chi2_stat
        rows[f"{prefix}_cox_p"] = p_value
        z_columns.append(f"{prefix}_cox_z")

    if not z_columns:
        raise ValueError("No dataset had enough train survival events for Cox gene selection.")

    z_matrix = rows[z_columns].to_numpy(dtype=np.float64)
    available = np.isfinite(z_matrix)
    rows["meta_n_datasets"] = available.sum(axis=1)
    signed_z = np.nan_to_num(z_matrix, nan=0.0)
    rows["meta_cox_z"] = signed_z.sum(axis=1) / np.sqrt(np.clip(rows["meta_n_datasets"].to_numpy(), 1, None))
    rows["meta_abs_z"] = rows["meta_cox_z"].abs()
    rows["meta_cox_p"] = 2.0 * norm.sf(rows["meta_abs_z"])
    rows["selection_direction"] = np.where(rows["meta_cox_z"].ge(0), "higher_expression_higher_risk", "higher_expression_lower_risk")

    rows = rows.sort_values(["meta_cox_p", "meta_abs_z", "gene_symbol"], ascending=[True, False, True]).reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    return rows


def build_literature_gene_table(available_genes: set[str], ranked_genes: pd.DataFrame) -> pd.DataFrame:
    records = []
    seen = set()
    rank_lookup = ranked_genes.set_index("gene_symbol").to_dict(orient="index")
    for category, genes in PDAC_LITERATURE_GENE_SETS.items():
        for gene in genes:
            gene = gene.upper()
            if gene in seen:
                continue
            seen.add(gene)
            ranking = rank_lookup.get(gene, {})
            records.append(
                {
                    "gene_symbol": gene,
                    "category": category,
                    "available_in_common_rnaseq": gene in available_genes,
                    "survival_cox_rank": ranking.get("rank"),
                    "meta_cox_z": ranking.get("meta_cox_z"),
                    "meta_cox_p": ranking.get("meta_cox_p"),
                    "selection_direction": ranking.get("selection_direction"),
                }
            )
    table = pd.DataFrame(records)
    table = table.sort_values(
        ["available_in_common_rnaseq", "survival_cox_rank", "gene_symbol"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    table.insert(0, "literature_order", np.arange(1, len(table) + 1))
    return table


def build_literature_guided_ranking(ranked_genes: pd.DataFrame, literature_table: pd.DataFrame) -> pd.DataFrame:
    curated_genes = (
        literature_table[literature_table["available_in_common_rnaseq"]]
        .sort_values(["survival_cox_rank", "gene_symbol"], na_position="last")["gene_symbol"]
        .drop_duplicates()
        .tolist()
    )
    curated_set = set(curated_genes)
    cox_genes = [gene for gene in ranked_genes["gene_symbol"].tolist() if gene not in curated_set]
    ordered_genes = curated_genes + cox_genes

    guided = ranked_genes.set_index("gene_symbol").loc[ordered_genes].reset_index()
    guided["is_literature_curated"] = guided["gene_symbol"].isin(curated_set)
    guided = guided.merge(
        literature_table[["gene_symbol", "category"]],
        on="gene_symbol",
        how="left",
    )
    guided = guided.rename(columns={"rank": "survival_cox_rank"})
    guided.insert(0, "rank", np.arange(1, len(guided) + 1))
    return guided


def save_selected_features(
    matrices: dict[str, pd.DataFrame],
    ranked_genes: pd.DataFrame,
    split_df: pd.DataFrame,
    n_genes: int,
    selection_name_prefix: str,
    output_prefix: str,
    selection_method: str,
) -> dict[str, object]:
    selected = ranked_genes.head(n_genes).copy()
    selected_genes = selected["gene_symbol"].tolist()
    out_dir = SELECTED_RNASEQ_DST_PATH / f"{selection_name_prefix}_{n_genes}"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_gene_path = out_dir / f"selected_genes_{selection_name_prefix}_{n_genes}.csv"
    selected.to_csv(selected_gene_path, index=False)
    split_df.to_csv(out_dir / "gene_selection_split.csv", index=False)

    dataset_summaries = []
    for dataset, matrix in matrices.items():
        dataset_dir = out_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        selected_matrix = matrix.reindex(columns=selected_genes)
        matrix_path = dataset_dir / f"matrix_rnaseq_zscore_{selection_name_prefix}_{n_genes}.csv"
        selected_matrix.to_csv(matrix_path)

        for case_id, row in selected_matrix.iterrows():
            values = row.to_numpy(dtype=np.float32)
            feature_path = dataset_dir / f"{case_id}_rnaseq_{output_prefix}_{n_genes}.npy"
            np.save(feature_path, values)

            metadata = {
                "dataset": dataset,
                "case_id": case_id,
                "feature_type": f"rnaseq_zscore_{selection_name_prefix}",
                "selection_method": selection_method,
                "n_genes": int(n_genes),
                "gene_file": selected_gene_path.as_posix(),
                "matrix_file": matrix_path.as_posix(),
                "feature_file": feature_path.as_posix(),
                "feature_dtype": "float32",
            }
            with (dataset_dir / f"{case_id}_rnaseq_{output_prefix}_{n_genes}.json").open("w") as f:
                json.dump(metadata, f, indent=2)

        dataset_summary = {
            "dataset": dataset,
            "n_cases": int(selected_matrix.shape[0]),
            "n_genes": int(selected_matrix.shape[1]),
            "matrix_file": matrix_path.as_posix(),
            "case_feature_dir": dataset_dir.as_posix(),
            "missing_values": int(selected_matrix.isna().sum().sum()),
        }
        dataset_summaries.append(dataset_summary)

    summary = {
        "selection_name": f"{selection_name_prefix}_{n_genes}",
        "selection_method": selection_method,
        "n_genes": int(n_genes),
        "selected_gene_file": selected_gene_path.as_posix(),
        "output_dir": out_dir.as_posix(),
        "datasets": dataset_summaries,
    }
    with (out_dir / "selection_metadata.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_feature_selection(
    n_genes_list: list[int],
    split_path: Path | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    zscore_matrices = load_rnaseq_zscore_matrices()
    split_df = load_or_build_split(split_path=split_path, seed=seed)
    ranked_genes = rank_genes_by_train_survival_cox(zscore_matrices, split_df)
    available_genes = set(ranked_genes["gene_symbol"])
    literature_table = build_literature_gene_table(available_genes=available_genes, ranked_genes=ranked_genes)
    literature_guided_genes = build_literature_guided_ranking(ranked_genes=ranked_genes, literature_table=literature_table)

    SELECTED_RNASEQ_DST_PATH.mkdir(parents=True, exist_ok=True)
    FEATURE_SELECTION_RESULT_PATH.mkdir(parents=True, exist_ok=True)

    ranked_path = FEATURE_SELECTION_RESULT_PATH / "rnaseq_gene_survival_cox_ranking.csv"
    ranked_genes.to_csv(ranked_path, index=False)
    literature_table.to_csv(FEATURE_SELECTION_RESULT_PATH / "rnaseq_pdac_literature_curated_genes.csv", index=False)
    literature_guided_genes.to_csv(FEATURE_SELECTION_RESULT_PATH / "rnaseq_gene_literature_guided_survival_ranking.csv", index=False)
    split_df.to_csv(FEATURE_SELECTION_RESULT_PATH / "rnaseq_gene_selection_split.csv", index=False)

    summaries = []
    for n_genes in n_genes_list:
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive: {n_genes}")
        if n_genes > len(ranked_genes):
            raise ValueError(f"n_genes={n_genes} exceeds available genes={len(ranked_genes)}")
        summaries.append(
            save_selected_features(
                zscore_matrices,
                ranked_genes,
                split_df,
                n_genes,
                selection_name_prefix="top_survival_cox",
                output_prefix="top_survival_cox",
                selection_method="train_split_univariate_cox_score_meta_analysis",
            )
        )
        summaries.append(
            save_selected_features(
                zscore_matrices,
                literature_guided_genes,
                split_df,
                n_genes,
                selection_name_prefix="top_literature_guided_survival_cox",
                output_prefix="top_literature_guided_survival_cox",
                selection_method="pdac_literature_curated_genes_prioritized_then_train_split_cox_ranking",
            )
        )

    summary_df = pd.DataFrame(
        {
            "selection_name": s["selection_name"],
            "selection_method": s["selection_method"],
            "n_genes": s["n_genes"],
            "selected_gene_file": s["selected_gene_file"],
            "output_dir": s["output_dir"],
        }
        for s in summaries
    )
    summary_df.to_csv(FEATURE_SELECTION_RESULT_PATH / "rnaseq_feature_selection_summary.csv", index=False)
    return literature_guided_genes, summary_df, split_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select compact survival-associated RNA-seq gene features for M3/M4.")
    parser.add_argument(
        "--n-genes",
        nargs="+",
        type=int,
        default=[1000, 1500, 2000],
        help="Number of top survival-associated genes to export.",
    )
    parser.add_argument("--split-path", type=Path, default=None, help="Optional case split CSV. Gene ranking uses train split only.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only when a split file is not available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked_genes, summary_df, split_df = run_feature_selection(args.n_genes, split_path=args.split_path, seed=args.seed)
    print("split:")
    print(split_df.groupby(["split", "dataset"])["case_id"].count().unstack(fill_value=0).to_string())
    print("\nranked genes:", ranked_genes.shape)
    print(ranked_genes.head(20).to_string(index=False))
    print("\nselected feature sets:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
