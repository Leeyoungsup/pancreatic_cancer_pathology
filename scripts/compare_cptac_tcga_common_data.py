from pathlib import Path

import pandas as pd


DATA_PATH = Path("../../data")
PROJECT_DATA_PATH = DATA_PATH / "pancreatic_cancer_pathology"
RAW_PATH = PROJECT_DATA_PATH / "raw"
TCGA_PATH = RAW_PATH / "TCGA_PAAD"
CPTAC_PATH = RAW_PATH / "CPTAC_PDAC"
OUTPUT_DIR = Path("outputs/data_verification/common_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_gene_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    symbol = str(value).strip()
    if symbol == "" or symbol.startswith("N_"):
        return None
    return symbol.upper()


def load_tcga_rna_gene_table() -> pd.DataFrame:
    rnaseq_manifest = pd.read_csv(
        TCGA_PATH / "TCGA_PAAD_matched" / "tcga_paad_matched_rnaseq_files.csv"
    )
    first_file_name = rnaseq_manifest["file_name"].iloc[0]
    first_file_path = TCGA_PATH / "RNA_SEQ_STAR_COUNTS" / first_file_name
    if not first_file_path.exists():
        matches = list((TCGA_PATH / "RNA_SEQ_STAR_COUNTS").rglob(first_file_name))
        if len(matches) == 0:
            raise FileNotFoundError(f"TCGA RNA-seq file not found: {first_file_name}")
        first_file_path = matches[0]

    gene_df = pd.read_csv(first_file_path, sep="\t", comment="#")
    gene_df["gene_symbol"] = gene_df["gene_name"].map(normalize_gene_symbol)
    gene_df = gene_df[gene_df["gene_symbol"].notna()].copy()
    return gene_df[["gene_id", "gene_name", "gene_type", "gene_symbol"]].drop_duplicates()


def load_cptac_gene_set(path: Path) -> set[str]:
    genes = pd.read_csv(path, sep="\t", usecols=[0]).iloc[:, 0]
    return {g for g in genes.map(normalize_gene_symbol) if g is not None}


def build_harmonized_clinical_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tcga = pd.read_csv(
        TCGA_PATH
        / "TCGA_PAAD_matched"
        / "tcga_paad_matched_patient_table_dx_one_per_patient.csv"
    )
    cptac = pd.read_csv(CPTAC_PATH / "matched" / "cptac_pda_matched_case_summary.csv")

    tcga_h = pd.DataFrame(
        {
            "cohort": "TCGA_PAAD",
            "subject_id": tcga["patient_id"],
            "age_years": pd.to_numeric(tcga["age_at_diagnosis"], errors="coerce") / 365.25,
            "sex": tcga["gender"].astype(str).str.lower(),
            "race": tcga["race"],
            "vital_status": tcga["vital_status"],
            "os_time_days": pd.to_numeric(tcga["OS_time"], errors="coerce"),
            "os_event": pd.to_numeric(tcga["OS_event"], errors="coerce"),
            "diagnosis": tcga["primary_diagnosis"],
            "pathologic_stage": tcga["ajcc_pathologic_stage"],
            "pathologic_t": tcga["ajcc_pathologic_t"],
            "pathologic_n": tcga["ajcc_pathologic_n"],
            "pathologic_m": tcga["ajcc_pathologic_m"],
            "tumor_grade": tcga["tumor_grade"],
            "has_wsi": tcga["selected_wsi_file_name"].notna(),
            "has_rnaseq": pd.to_numeric(tcga["n_rnaseq_files"], errors="coerce").fillna(0).gt(0),
        }
    )

    cptac_h = pd.DataFrame(
        {
            "cohort": "CPTAC_PDAC",
            "subject_id": cptac["case_id"],
            "age_years": pd.to_numeric(cptac["age"], errors="coerce"),
            "sex": cptac["sex"].astype(str).str.lower(),
            "race": cptac["race"],
            "vital_status": cptac["vital_status"],
            "os_time_days": pd.to_numeric(cptac["follow_up_days"], errors="coerce"),
            "os_event": cptac["vital_status"].astype(str).str.lower().eq("deceased").astype(int),
            "diagnosis": cptac["histology_diagnosis"],
            "pathologic_stage": cptac["tumor_stage_pathological"],
            "pathologic_t": cptac["pathologic_staging_primary_tumor_pt"],
            "pathologic_n": cptac["pathologic_staging_regional_lymph_nodes_pn"],
            "pathologic_m": cptac["pathologic_staging_distant_metastasis_pm"],
            "tumor_grade": pd.NA,
            "has_wsi": pd.to_numeric(cptac["n_wsi_series"], errors="coerce").fillna(0).gt(0),
            "has_rnaseq": cptac["case_id"].isin(
                pd.read_csv(CPTAC_PATH / "matched" / "rna_cases.csv")["case_id"]
            ),
        }
    )

    variable_map = pd.DataFrame(
        [
            ("subject_id", "patient_id", "case_id", "환자/케이스 ID"),
            ("age_years", "age_at_diagnosis / 365.25", "age", "진단 시 연령"),
            ("sex", "gender", "sex", "성별"),
            ("race", "race", "race", "인종"),
            ("vital_status", "vital_status", "vital_status", "생존 상태"),
            ("os_time_days", "OS_time", "follow_up_days", "전체생존/추적 기간, days"),
            ("os_event", "OS_event", "vital_status == Deceased", "사망 event"),
            ("diagnosis", "primary_diagnosis", "histology_diagnosis", "진단명/조직형"),
            ("pathologic_stage", "ajcc_pathologic_stage", "tumor_stage_pathological", "병리 stage"),
            ("pathologic_t", "ajcc_pathologic_t", "pathologic_staging_primary_tumor_pt", "병리 T stage"),
            ("pathologic_n", "ajcc_pathologic_n", "pathologic_staging_regional_lymph_nodes_pn", "병리 N stage"),
            ("pathologic_m", "ajcc_pathologic_m", "pathologic_staging_distant_metastasis_pm", "병리 M stage"),
            ("tumor_grade", "tumor_grade", None, "TCGA에만 있음"),
            ("has_wsi", "selected_wsi_file_name", "n_wsi_series", "병리 WSI 보유 여부"),
            ("has_rnaseq", "n_rnaseq_files", "rna_cases.csv", "RNA-seq 보유 여부"),
        ],
        columns=["harmonized_variable", "tcga_source", "cptac_source", "note"],
    )
    variable_map["available_in_both"] = variable_map["tcga_source"].notna() & variable_map[
        "cptac_source"
    ].notna()
    return tcga_h, cptac_h, variable_map


def summarize_clinical_completeness(tcga_h: pd.DataFrame, cptac_h: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([tcga_h, cptac_h], ignore_index=True)
    rows = []
    for cohort, df in combined.groupby("cohort"):
        for col in df.columns:
            if col == "cohort":
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "variable": col,
                    "n": len(df),
                    "missing_count": int(df[col].isna().sum()),
                    "missing_rate": float(df[col].isna().mean()),
                    "n_unique": int(df[col].nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def main() -> dict[str, pd.DataFrame]:
    tcga_h, cptac_h, variable_map = build_harmonized_clinical_tables()
    clinical_completeness = summarize_clinical_completeness(tcga_h, cptac_h)

    tcga_h.to_csv(OUTPUT_DIR / "tcga_common_clinical_harmonized.csv", index=False)
    cptac_h.to_csv(OUTPUT_DIR / "cptac_common_clinical_harmonized.csv", index=False)
    variable_map.to_csv(OUTPUT_DIR / "clinical_common_variable_map.csv", index=False)
    clinical_completeness.to_csv(OUTPUT_DIR / "clinical_common_completeness.csv", index=False)

    tcga_gene_table = load_tcga_rna_gene_table()
    tcga_rna_genes = set(tcga_gene_table["gene_symbol"])
    tcga_rna_protein_coding_genes = set(
        tcga_gene_table.loc[tcga_gene_table["gene_type"].eq("protein_coding"), "gene_symbol"]
    )

    cptac_omics_paths = {
        "cptac_rna": CPTAC_PATH / "omics" / "rna_tumor_rsem_uq_log2.cct",
        "cptac_proteome": CPTAC_PATH / "omics" / "proteomics_tumor_gene_level.cct",
        "cptac_phosphoproteome": CPTAC_PATH
        / "omics"
        / "phosphoproteomics_tumor_gene_level.cct",
    }
    gene_sets = {
        "tcga_rna_all": tcga_rna_genes,
        "tcga_rna_protein_coding": tcga_rna_protein_coding_genes,
    }
    for name, path in cptac_omics_paths.items():
        gene_sets[name] = load_cptac_gene_set(path)

    comparisons = {
        "tcga_rna_all__cptac_rna": gene_sets["tcga_rna_all"] & gene_sets["cptac_rna"],
        "tcga_rna_protein_coding__cptac_rna": gene_sets["tcga_rna_protein_coding"]
        & gene_sets["cptac_rna"],
        "tcga_rna_all__cptac_proteome": gene_sets["tcga_rna_all"] & gene_sets["cptac_proteome"],
        "tcga_rna_all__cptac_phosphoproteome": gene_sets["tcga_rna_all"]
        & gene_sets["cptac_phosphoproteome"],
        "tcga_rna_all__cptac_rna__proteome": gene_sets["tcga_rna_all"]
        & gene_sets["cptac_rna"]
        & gene_sets["cptac_proteome"],
        "tcga_rna_all__cptac_rna__proteome__phosphoproteome": gene_sets["tcga_rna_all"]
        & gene_sets["cptac_rna"]
        & gene_sets["cptac_proteome"]
        & gene_sets["cptac_phosphoproteome"],
    }

    gene_set_summary = pd.DataFrame(
        [{"gene_set": name, "n_genes": len(genes)} for name, genes in gene_sets.items()]
        + [
            {"gene_set": f"COMMON:{name}", "n_genes": len(genes)}
            for name, genes in comparisons.items()
        ]
    )
    gene_set_summary.to_csv(OUTPUT_DIR / "omics_gene_set_summary.csv", index=False)

    for name, genes in comparisons.items():
        out = pd.DataFrame({"gene_symbol": sorted(genes)})
        out.to_csv(OUTPUT_DIR / f"common_genes_{name}.csv", index=False)

    return {
        "clinical_variable_map": variable_map,
        "clinical_completeness": clinical_completeness,
        "gene_set_summary": gene_set_summary,
        "tcga_clinical": tcga_h,
        "cptac_clinical": cptac_h,
    }


if __name__ == "__main__":
    results = main()
    print("Saved outputs to:", OUTPUT_DIR)
    print("\nClinical common variable map")
    print(results["clinical_variable_map"].to_string(index=False))
    print("\nOmics gene set summary")
    print(results["gene_set_summary"].to_string(index=False))
