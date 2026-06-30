# Method

## 1. 데이터 선정 방식

본 연구는 췌장암 생존예측 모델 개발을 위해 TCGA-PAAD와 CPTAC-PDAC 코호트를 사용하였다. 두 데이터셋은 병리 whole-slide image(WSI), 임상 생존 정보, RNA-seq 데이터가 모두 존재하는 케이스를 중심으로 구성하였다. 원본 데이터는 수정하지 않고, 분석 가능 여부를 확인한 뒤 전처리 결과를 `../../data/pancreatic_cancer_pathology/dst/` 하위에 별도로 저장하였다.

### 1.1 TCGA-PAAD 코호트

TCGA-PAAD에서는 diagnostic slide(`DX`)에 해당하는 SVS 형식의 WSI와 RNA-seq STAR counts 파일이 모두 존재하는 환자를 선별하였다. 환자 단위 분석을 위해 동일 환자에서 여러 WSI가 존재하는 경우 대표 diagnostic WSI 1개를 선택하였다. 생존 라벨은 `vital_status`, `days_to_death`, `days_to_last_follow_up` 정보를 이용하여 overall survival(OS) 기준으로 정리하였다.

최종 TCGA-PAAD 후보 코호트는 `outputs/data_verification/tcga_survival_cohort_candidate.csv`에 저장하였으며, 총 160명의 환자가 포함되었다.

### 1.2 CPTAC-PDAC 코호트

CPTAC-PDAC에서는 H&E WSI DICOM series, 임상 생존 정보, RNA-seq 데이터가 모두 존재하는 케이스를 선별하였다. WSI series 중 병리 조직 분석에 적합한 `HE tumor` series를 우선적으로 사용하였다. 생존 라벨은 follow-up 기간과 vital status를 이용하여 TCGA와 동일하게 OS 기준으로 정리하였다.

최종 CPTAC-PDAC 후보 코호트는 `outputs/data_verification/cptac_survival_cohort_candidate.csv`에 저장하였으며, 총 140명의 케이스가 포함되었다.

### 1.3 데이터 출처 기관 및 지역 정보

TCGA-PAAD의 경우 환자 barcode의 두 번째 항목은 TCGA tissue source site(TSS) code를 의미한다. 예를 들어 `TCGA-IB-7645`의 TSS code는 `IB`이며, 이는 GDC에서 제공하는 TCGA Tissue Source Site code table을 통해 tissue source site로 매핑할 수 있다. 따라서 TCGA-PAAD에서는 각 case별 병리 slide의 제공기관 또는 tissue source site를 TSS code 수준에서 확인할 수 있다. 단, TSS는 실제 진료 병원명과 항상 동일한 개념은 아니며, tissue procurement 기관 또는 biobank가 포함될 수 있다.

본 연구에 포함된 TCGA-PAAD 160명은 24개 TSS code에서 유래하였다. 주요 TSS 분포는 다음과 같다.

- `IB`: Alberta Health Services, 37 cases
- `HZ`: International Genomics Consortium, 28 cases
- `FB`: Asterand, 16 cases
- `2J`: Mayo Clinic, 15 cases
- `3A`: Moffitt Cancer Center, 13 cases
- `F2`: UNC, 8 cases
- `HV`: National Cancer Center Korea, 8 cases
- `2L`: Technical University of Munich, 6 cases
- `US`: Garvan Institute of Medical Research, 6 cases

그 외 TSS code는 `LB`, `XD`, `Q3`, `RB`, `S4`, `3E`, `H6`, `H8`, `L1`, `OE`, `PZ`, `RL`, `XN`, `YB`, `YY`, `Z5`로 구성되었다.

CPTAC-PDAC의 경우 현재 확보한 matched clinical table에는 병원 또는 기관명에 해당하는 컬럼이 포함되어 있지 않았다. 따라서 case별 병원 단위 출처는 확인하지 않았고, 공개 clinical metadata에서 제공되는 `participant_country`를 지역 정보로 사용하였다. CPTAC-PDAC 140 cases의 국가 분포는 다음과 같다.

- Poland: 45 cases
- United States: 28 cases
- China: 21 cases
- Canada: 19 cases
- Russia: 11 cases
- Other: 10 cases
- Bulgaria, South Wales, Denmark, Serbia, India, Iraq: each 1 case

따라서 본 연구에서는 TCGA-PAAD의 출처 정보는 TSS code 기반 source site 수준으로, CPTAC-PDAC의 출처 정보는 participant country 수준으로 정리하였다.

### 1.4 공통 임상 변수 및 공통 유전자 선정

두 데이터셋의 임상 변수는 변수명과 값 표현 방식이 서로 다르기 때문에, 분석 전에 공통 schema로 정규화하였다. 공통 임상 변수는 다음 항목을 중심으로 정리하였다.

- `age_years`
- `sex`
- `race`
- `vital_status`
- `os_time_days`
- `os_event`
- `diagnosis`
- `pathologic_stage`
- `pathologic_t`
- `pathologic_n`
- `pathologic_m`
- `tumor_grade`
- `has_wsi`
- `has_rnaseq`

`os_event`는 사망 여부를 의미하며, 재발 여부가 아니다. 사망한 경우 1, 생존 또는 censoring된 경우 0으로 정의하였다.

Omics 데이터는 TCGA RNA-seq protein-coding gene과 CPTAC RNA-seq gene symbol의 교집합을 사용하였다. 최종 공통 유전자는 18,879개이며, 목록은 `outputs/data_verification/common_data/common_genes_tcga_rna_protein_coding__cptac_rna.csv`에 저장하였다.

## 2. 학습을 위한 데이터 전처리

전처리는 `data_preprocessing.ipynb`에서 수행하였다. 학습 입력은 병리 이미지 tile, 표준화된 임상 JSON, RNA-seq feature로 구성하였다.

### 2.1 병리 이미지 전처리

WSI는 데이터셋별 원본 형식이 다르기 때문에 TCGA와 CPTAC를 분리하여 처리하였다. TCGA-PAAD는 SVS WSI를 사용하였고, CPTAC-PDAC는 DICOM WSI series를 사용하였다.

모든 WSI는 target resolution `MPP = 1.0`에서 `1024 x 1024` pixel tile로 분할하였다. 원본 slide의 native MPP와 pyramid level을 확인한 뒤 target MPP에 가장 가까운 level에서 tile을 읽고, 최종 tile 크기를 1024 x 1024로 맞추었다. 배경 tile 저장을 줄이기 위해 low-resolution tissue mask를 먼저 생성하고, tissue 비율이 낮은 흰 배경 영역은 후보 tile 단계에서 제외하였다. 최종 tile 저장 기준은 tissue 비율 threshold `0.15`를 사용하였다.

TCGA의 경우 slide pyramid level을 이용하여 target MPP에 가까운 level에서 tile을 읽었다. CPTAC의 경우 DICOM 전체 pixel array를 메모리에 올리지 않고 필요한 frame만 부분 decoding하여 tile을 저장하였다. 전처리 속도를 위해 case 단위 병렬 처리를 적용하였다.

이미지 tile은 다음 경로에 저장하였다.

- TCGA-PAAD: `../../data/pancreatic_cancer_pathology/dst/Image/TCGA_PAAD/{case_id}/`
- CPTAC-PDAC: `../../data/pancreatic_cancer_pathology/dst/Image/CPTAC_PDAC/{case_id}/`

각 데이터셋별 tile 처리 요약은 다음 파일에 저장하였다.

- `../../data/pancreatic_cancer_pathology/dst/Image/TCGA_PAAD/tile_summary.csv`
- `../../data/pancreatic_cancer_pathology/dst/Image/CPTAC_PDAC/tile_summary.csv`

### 2.2 임상 데이터 전처리

TCGA와 CPTAC의 임상 데이터는 공통 변수 schema에 맞춰 정규화한 뒤 케이스별 JSON 파일로 저장하였다. 성별, 인종, vital status, 병리 병기 등의 범주형 변수는 데이터셋 간 표현 차이를 줄이기 위해 표준화하였다. 생존 시간은 `os_time_days`로 통일하였고, 생존 event는 `os_event`로 저장하였다.

임상 JSON은 다음 경로에 저장하였다.

- TCGA-PAAD: `../../data/pancreatic_cancer_pathology/dst/Clinical/TCGA_PAAD/{case_id}_clinical.json`
- CPTAC-PDAC: `../../data/pancreatic_cancer_pathology/dst/Clinical/CPTAC_PDAC/{case_id}_clinical.json`

현재 저장된 임상 JSON은 TCGA-PAAD 160개, CPTAC-PDAC 140개이다. 각 JSON에는 모델 입력에 사용할 표준화된 `clinical` 항목과 추적 가능성을 위한 `clinical_raw` 항목을 함께 저장하였다.

### 2.3 RNA-seq 전처리

RNA-seq 데이터는 두 데이터셋에서 측정 및 저장 형식이 다르기 때문에 각각 전처리한 뒤 공통 gene set 기준으로 정렬하였다.

TCGA-PAAD는 STAR counts 결과의 `fpkm_uq_unstranded` 값을 사용하였다. 각 gene expression 값은 `log2(FPKM-UQ + 1)`로 변환하였다. 동일 gene symbol이 여러 row에 존재하는 경우 protein-coding gene을 우선하고, 같은 symbol 내 값은 평균으로 collapse하였다.

CPTAC-PDAC는 `rna_tumor_rsem_uq_log2.cct` 파일을 사용하였다. 해당 파일은 RSEM upper-quartile normalized expression의 log2 scale 값으로 제공되므로 추가 log 변환 없이 사용하였다. 동일 gene symbol이 여러 row에 존재하는 경우 평균으로 collapse하였다.

두 데이터셋 모두 최종적으로 18,879개 공통 gene 순서로 case x gene matrix를 구성하였다. 이후 각 데이터셋 내부에서 gene별 평균과 표준편차를 계산하여 z-score 정규화를 수행하였다.

RNA-seq 결과는 다음 경로에 저장하였다.

- TCGA-PAAD: `../../data/pancreatic_cancer_pathology/dst/RNAseq/TCGA_PAAD/`
- CPTAC-PDAC: `../../data/pancreatic_cancer_pathology/dst/RNAseq/CPTAC_PDAC/`

주요 산출물은 다음과 같다.

- `matrix_common_genes_log2_fpkm_uq.csv`: TCGA case x gene log2 expression matrix
- `matrix_common_genes_log2_rsem_uq.csv`: CPTAC case x gene log2 expression matrix
- `matrix_common_genes_zscore.csv`: gene별 z-score 정규화 matrix
- `genes_common_protein_coding.csv`: 사용한 공통 gene 목록
- `zscore_reference.csv`: gene별 평균과 표준편차
- `{case_id}_rnaseq_zscore.npy`: 케이스별 RNA-seq feature vector
- `{case_id}_rnaseq.json`: 케이스별 RNA-seq metadata
- `rnaseq_json_summary.csv`: 케이스별 RNA-seq 저장 요약

최종 RNA-seq feature matrix의 크기는 TCGA-PAAD 160 x 18,879, CPTAC-PDAC 140 x 18,879이다. 결측값은 두 데이터셋 모두 0개로 확인하였다.

## 3. 최종 학습 데이터 구성

최종 모델 학습에서는 case ID를 기준으로 세 종류의 입력을 연결한다.

- 병리 이미지 tile: `dst/Image/{dataset}/{case_id}/`
- 임상 정보: `dst/Clinical/{dataset}/{case_id}_clinical.json`
- RNA-seq feature: `dst/RNAseq/{dataset}/{case_id}_rnaseq_zscore.npy`

예측 라벨은 임상 JSON의 `os_time_days`와 `os_event`를 사용한다. 따라서 본 연구의 기본 task는 overall survival prediction이며, recurrence-free survival이나 disease-free survival이 아니다.

데이터 분할 시에는 tile 단위가 아니라 case 단위로 train, validation, test를 분리하여 동일 환자에서 생성된 tile이 서로 다른 split에 섞이지 않도록 한다. 이는 WSI tile 기반 학습에서 발생할 수 있는 data leakage를 방지하기 위한 기준이다.

## 4. 모델 구성 및 비교 실험

본 연구의 핵심 목적은 병리 WSI 기반 생존예측 모델을 구축하고, 최소 임상정보와 RNA-seq 정보가 병리 이미지 기반 예측 성능에 추가적인 기여를 하는지 평가하는 것이다. 따라서 단독 clinical model이나 단독 RNA-seq model은 주요 비교 모델로 사용하지 않고, 병리 이미지를 중심으로 입력 조합을 변화시키는 네 가지 모델을 구성한다.

모든 모델의 예측 목표는 동일하게 overall survival(OS)이며, label은 `os_time_days`와 `os_event`를 사용한다. `os_event`는 사망 event를 의미하며 재발 event가 아니다.

### 4.1 입력 변수 제외 기준

다음 변수들은 생존 예후와 직접적으로 강하게 연결된 병리학적 요약 정보이므로 기본 모델 입력에서는 제외한다.

- `pathologic_stage`
- `pathologic_t`
- `pathologic_n`
- `pathologic_m`
- `tumor_grade`

이 변수들은 WSI와 RNA-seq가 예후를 예측하는 독립적인 정보를 갖는지 평가하려는 연구 목적에서 shortcut feature로 작동할 수 있다. 따라서 기본 학습 입력에서는 제외하고, 필요 시 추후 subgroup analysis 또는 보조적인 ablation 분석에서만 별도로 검토한다.

기본 clinical 입력은 최소 임상정보로 제한한다.

- `age_years`
- `sex`

`race`는 데이터셋별 인구집단 구성 차이와 bias 가능성을 고려하여 기본 모델에서는 제외하고, 필요 시 민감도 분석에서만 사용한다.

### 4.2 비교 모델

| Model | 입력 데이터 | 목적 |
|---|---|---|
| M1 | Pathology only | WSI tile 기반 병리 이미지 정보만으로 OS 예측이 가능한지 평가 |
| M2 | Pathology + basic clinical | 병리 이미지에 나이와 성별을 추가했을 때 예측 성능이 개선되는지 평가 |
| M3 | Pathology + RNA-seq | 병리 형태학적 정보와 분자 정보가 상호보완적인지 평가 |
| M4 | Pathology + RNA-seq + basic clinical | 병리 이미지, 분자 정보, 최소 임상정보를 결합한 최종 multimodal model |

M1은 병리 이미지 자체의 예후 예측력을 평가하는 기준 모델이다. WSI tile에서 feature를 추출한 뒤 case-level MIL aggregation을 통해 환자 단위 risk score를 예측한다.

M2는 M1에 `age_years`와 `sex`를 추가한 모델이다. 이 모델은 실제 임상 적용 상황에서 쉽게 확보 가능한 최소 임상정보가 병리 이미지 기반 모델을 얼마나 보완하는지 평가한다.

M3는 병리 이미지 feature와 RNA-seq feature를 결합한 모델이다. RNA-seq 입력은 공통 gene 기반 z-score feature를 사용하되, 전체 18,879개 gene을 직접 사용하는 대신 학습 단계에서 feature selection 또는 embedding projection을 적용할 수 있다. 이 모델은 형태학적 phenotype과 molecular profile의 상호보완성을 평가하기 위한 핵심 비교 모델이다.

M4는 병리 이미지, RNA-seq, 최소 임상정보를 모두 사용하는 최종 multimodal model이다. 각 modality별 encoder를 통해 embedding을 생성한 뒤 fusion layer에서 결합하여 최종 risk score를 예측한다.

### 4.3 모델 구조 개요

현재 구현한 M1과 M2는 환자 단위 overall survival(OS) risk score를 예측하는 weakly supervised WSI survival model이다. 모델 입력은 case별 병리 tile image와 tile coordinate이며, M2에서는 여기에 최소 임상정보(`age_years`, `sex`)를 추가한다.

병리 이미지 branch는 pretrained pathology foundation model인 UNI 또는 UNI2-h를 tile-level encoder로 사용한다. 현재 실험에서는 foundation encoder를 freeze하고, 그 위의 tile fusion layer, latent tile risk head, risk-aware pooling layer, survival prediction head만 학습한다. 이는 TCGA-PAAD와 CPTAC-PDAC를 합친 후에도 학습 가능한 환자 수가 제한적이기 때문에, 대규모 backbone fine-tuning으로 인한 overfitting을 줄이기 위한 설정이다.

최종 출력은 환자 단위 scalar risk score이다.

```text
r_i = f_model(WSI_i)
```

여기서 `r_i`가 높을수록 더 짧은 생존 기간과 관련된 높은 위험도로 해석한다. 모델은 직접적인 시점별 사망확률을 출력하지 않고, Cox partial likelihood loss를 통해 survival time ranking을 학습한다.

RNA-seq branch는 M3/M4에서 사용할 계획이다. RNA-seq 입력은 공통 gene 기반 z-score vector를 사용하되, 전체 18,879개 gene을 직접 사용하는 대신 학습 단계에서 feature selection 또는 embedding projection을 적용할 수 있다. Feature selection을 사용할 경우 train split에서만 기준을 계산하여 validation/test leakage를 방지한다.

### 4.4 M1 Pathology-only Model

M1은 병리 WSI만 사용하는 pathology-only model이다. 각 case의 WSI를 bag으로 정의하고, slide 내부 tissue tile을 instance로 정의한다. 환자 `i`의 slide는 `N_i`개의 tile image와 coordinate로 구성된 bag `B_i = {(x_ij, c_ij)}_{j=1}^{N_i}`로 표현한다.

학습 시 한 slide에서 최대 512개 tile을 사용한다. Slide 내 tile 수가 512개를 초과하는 경우 매 epoch마다 random sampling을 수행하여 서로 다른 tile subset이 모델에 입력되도록 하였고, validation/test에서는 deterministic sampling을 적용한다. 각 tile은 1.0 MPP에서 생성된 1024 x 1024 image를 512 x 512로 resize하여 입력한다. 따라서 모델 입력 기준 effective resolution은 약 2.0 MPP이다. UNI2-h와 같이 patch size가 14인 ViT feature extractor를 사용하는 경우 symmetric padding을 적용하여 518 x 518로 맞춘다.

Tile-level image encoder는 frozen UNI/UNI2-h를 사용한다.

```text
z_ij = f_UNI(x_ij)
```

현재 main setting에서는 coordinate embedding을 비활성화하였다(`use_spatial_embedding=False`). Coordinate tensor는 tile sampling과 추후 ablation을 위해 유지하지만, 현재 prediction에는 사용하지 않는다. Coordinate embedding을 사용하는 ablation에서는 다음 6개 normalized coordinate feature를 MLP로 embedding한 뒤 image feature와 결합할 수 있다.

- `x_norm`
- `y_norm`
- `x_center_norm`
- `y_center_norm`
- `w_norm`
- `h_norm`

Frozen UNI/UNI2-h feature는 trainable tile fusion MLP를 통과한다.

```text
h_ij = f_fusion(z_ij)
```

M1의 핵심 aggregation module은 latent tile risk scoring과 risk-weighted/top-k pooling이다. 먼저 각 tile representation `h_ij`에 대해 tile-level latent risk score를 계산한다.

```text
u_ij = f_risk(h_ij)
ρ_ij = sigmoid(u_ij)
```

여기서 `ρ_ij`는 tile-level ground truth로 직접 supervision되는 값이 아니라, slide-level Cox loss를 통해 간접적으로 학습되는 survival-associated latent risk score이다. 따라서 이를 병리학적 악성도 label로 해석하지 않고, 생존예측 task에 의해 학습된 tile-level risk proxy로 해석한다.

Slide-level representation은 단순 평균이 아니라 전체 tile 분포와 고위험 tile feature를 함께 사용하여 구성한다.

```text
H_mean = mean_j(h_ij)
H_std = std_j(h_ij)
α_ij = softmax_j(u_ij)
H_risk = Σ_j α_ij h_ij
H_top10 = mean(h_ij | ρ_ij in top 10%)
H_top25 = mean(h_ij | ρ_ij in top 25%)
R_stats = [mean(ρ), std(ρ), max(ρ), top10_mean(ρ), top25_mean(ρ)]
H_i = [H_mean, H_std, H_risk, H_top10, H_top25, R_stats]
```

최종 survival head는 `H_i`로부터 환자 단위 scalar risk score를 출력한다.

```text
r_i = f_head(H_i)
```

M1의 trainable component는 tile fusion MLP, latent tile risk head, risk-weighted/top-k pooling module, survival prediction head이다. UNI/UNI2-h feature extractor는 현재 실험에서 freeze한다.

M1 구현 파일은 `scripts/models/m1_pathology_mil.py`이며, 주요 구성요소는 다음과 같다.

- `SpatialEmbedding`
- `LatentRiskTopKPooling`
- `PathologySpatialMIL`

### 4.5 M2 Pathology + Basic Clinical Model

M2는 M1의 pathology branch에 최소 임상정보를 결합한 multimodal model이다. M2의 목적은 병리 이미지 기반 representation에 기본 임상정보를 추가했을 때 생존예측 성능이 개선되는지 평가하는 것이다. 임상 입력은 `age_years`와 `sex`만 사용하며, 병리 병기, TNM stage, tumor grade는 모델 입력에서 제외한다.

Age는 training split의 평균과 표준편차를 이용하여 z-score 정규화한다. Sex는 binary one-hot encoding으로 변환한다. 따라서 환자 `i`의 clinical vector `u_i`는 다음 3개 변수로 구성된다.

- `age_years_z`
- `sex_male`
- `sex_female`

M2의 pathology branch는 M1과 동일하다. 즉, 각 tile image는 frozen UNI/UNI2-h encoder를 통과하고, tile fusion MLP와 latent tile risk scoring module을 거쳐 risk-weighted/top-k pooling 기반 slide-level pathology embedding `H_i`를 생성한다.

Clinical vector `u_i`는 별도의 clinical embedding MLP를 통해 `g_i = f_clinical(u_i)`로 변환한다. 이후 slide-level pathology embedding과 clinical embedding을 late fusion 방식으로 concatenate한다.

```text
H_i = RiskTopKPooling({f_fusion(f_UNI(x_ij))}_{j=1}^{N_i})
g_i = f_clinical(u_i)
q_i = [H_i, g_i]
r_i = f_head(q_i)
```

M2는 tile-level 단계에서는 clinical 정보를 사용하지 않고, slide-level pathology embedding이 생성된 뒤 clinical embedding을 결합한다. 이 late-fusion design은 tile-level latent risk scoring이 병리 image feature에 기반하여 계산되도록 하며, 이후 환자 단위 임상정보가 최종 risk prediction을 보정하도록 한다. M2의 trainable component는 M1의 trainable module에 clinical embedding MLP와 clinical-pathology fusion head가 추가된 형태이다.

M2 구현 파일은 `scripts/models/m2_pathology_clinical_mil.py`이며, 주요 구성요소는 다음과 같다.

- `SpatialEmbedding`
- `ClinicalEmbedding`
- `LatentRiskTopKPooling`
- `PathologyClinicalMIL`

### 4.6 학습 구조

M1과 M2는 동일한 survival label 구조와 학습 loop를 사용한다. 현재 실험에서는 TCGA-PAAD와 CPTAC-PDAC를 합친 combined cohort를 사용한다. 학습에는 24개월까지 생존 라벨을 확정할 수 있는 complete-24m case만 포함한다. 이 기준을 적용하면 TCGA-PAAD 112 cases와 CPTAC-PDAC 93 cases, 총 205 cases가 학습 후보가 된다.

데이터 분할은 case 단위로 수행하며, 동일 case에서 생성된 tile이 서로 다른 split에 섞이지 않도록 한다. Train, validation, test 비율은 6:2:2로 설정한다. Combined cohort에서 TCGA와 CPTAC의 비율 및 event 비율이 split 간 크게 달라지지 않도록 `dataset + os_event` 조합을 기준으로 stratified split을 적용한다.

Survival label은 `os_time_days`와 `os_event`를 사용한다. `os_event=1`은 사망 event가 관측된 경우이고, `os_event=0`은 censoring을 의미한다. 모델은 각 case에 대해 하나의 scalar risk score `r_i`를 출력하며, loss는 Cox partial likelihood의 negative log-likelihood로 계산한다.

```text
L_Cox = - Σ_{i:E_i=1} [r_i - log Σ_{j:T_j≥T_i} exp(r_j)]
```

Cox loss는 case batch 내부에서 계산한다. 현재 `CASE_BATCH_SIZE=16`으로 설정하여 여러 case의 risk set을 한 번에 구성한다. Optimizer는 AdamW를 사용하고, validation C-index 기준 `ReduceLROnPlateau` scheduler를 적용한다.

학습 중 train/validation loop 모두 `tqdm`으로 진행 상황을 표시한다. Progress bar에는 running Cox loss, C-index, median-risk high/low group의 hazard ratio(HR), log-rank p-value를 표시한다. Checkpoint는 validation C-index가 가장 높은 epoch을 best checkpoint로 저장한다.

평가 지표는 다음을 사용한다.

- Harrell's C-index
- Median-risk group Kaplan-Meier curve
- Log-rank test
- High-risk vs low-risk hazard ratio(HR) 및 95% confidence interval

Risk group은 validation 또는 test prediction의 median risk score를 기준으로 high-risk와 low-risk로 나눈다.

Checkpoint에는 trainable module의 state dict와 optimizer/scheduler state, training config, split 정보, clinical normalization 정보를 저장한다. Feature extractor인 UNI/UNI2-h weight는 frozen 상태이고 파일 크기가 크기 때문에 checkpoint에는 저장하지 않는다. 재개 시에는 UNI/UNI2-h를 먼저 다시 로드한 뒤 checkpoint를 `strict=False`로 로드한다.

## 5. 논문 작성용 요약 문장

We constructed a multi-modal pancreatic cancer cohort from TCGA-PAAD and CPTAC-PDAC by selecting cases with available H&E whole-slide images, RNA-seq profiles, and overall survival information. Whole-slide images were tiled at 1.0 microns per pixel into 1024 x 1024 patches after excluding low-tissue background regions using low-resolution tissue masks. Clinical variables were harmonized into a common schema across datasets and stored as case-level JSON files. RNA-seq profiles were transformed to log2-scale expression values, restricted to 18,879 common protein-coding genes, and standardized using gene-wise z-score normalization within each dataset. The final processed dataset consisted of 160 TCGA-PAAD cases and 140 CPTAC-PDAC cases with matched image tiles, clinical variables, RNA-seq features, and overall survival labels.

For pathology-based survival prediction, we developed weakly supervised WSI models using frozen UNI/UNI2-h tile features and a trainable latent tile-risk pooling module. Each WSI was represented as a bag of tissue tiles, and each tile was assigned a survival-associated latent risk score learned only from patient-level survival supervision. Slide-level representations were constructed by combining global tile feature statistics, risk-weighted feature pooling, top-risk tile feature pooling, and tile-risk distribution statistics. The pathology-only model (M1) predicted a patient-level scalar risk score from WSI tiles alone, whereas the pathology-clinical model (M2) fused the WSI representation with age and sex embeddings. Models were trained using Cox partial likelihood loss and evaluated using Harrell's C-index, Kaplan-Meier risk stratification, log-rank test, and high-risk versus low-risk hazard ratio.
