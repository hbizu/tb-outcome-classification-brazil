
> **Paper:** *Comparison of Tree-Based Machine Learning Models for Classification of Tuberculosis Outcomes in Brazil*  
> **Authors:** Heloísa de Almeida Pereira, Marcos Roberto Ribeiro, Ciniro Aparecido Leite Nametala
> **Affiliation:** Federal Institute of Minas Gerais (IFMG), Brazil  

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
   - [Source](#source)
   - [Data Dictionary](#data-dictionary)
   - [Variables Used in the Study](#variables-used-in-the-study)
   - [Target Variable](#target-variable)
4. [Notebooks](#notebooks)
   - [data\_analysis.ipynb](#data_analysisipynb)
   - [tuberculosis\_outcome\_classification.ipynb](#tuberculosis_outcome_classificationipynb)
5. [Methodology](#methodology)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [Class Imbalance Handling](#class-imbalance-handling)
   - [Model Configuration](#model-configuration)
   - [Evaluation Strategy](#evaluation-strategy)
6. [Results Summary](#results-summary)
7. [Requirements](#requirements)
8. [How to Run](#how-to-run)
---

## Overview

Tuberculosis remains the leading cause of death from a single infectious agent worldwide (WHO, 2025), and its persistence in Brazil is strongly linked to socioeconomic vulnerability and gaps in health system management.

This project evaluates four tree-based ensemble classifiers:

| Model | Library |
|---|---|
| Random Forest | `scikit-learn` |
| XGBoost | `xgboost` |
| CatBoost | `catboost` |
| LightGBM | `lightgbm` |

The models are applied to **53,656 epidemiological records** from Minas Gerais, Brazil (2010–2024), classifying four treatment outcomes: **Cure**, **Abandonment**, **Death from Tuberculosis**, and **Death from Other Causes**. The study investigates the effect of SMOTE-based class balancing and analyzes the contribution of clinical versus socioeconomic features.

---

## Repository Structure

```

├── data/
│   └── dados_tuberculose_mg.xlsx              # Source dataset (not included — see Dataset section)
├── data_analysis.ipynb                        # Exploratory data analysis and visualizations
├── tuberculosis_outcome_classification.ipynb  # Preprocessing, model training, and evaluation
├── figures/
│   ├── architecture_diagram.pdf
│   ├── class_distribution.pdf
│   ├── temporal_forward_validation.pdf
│   ├── confusion_matrix_rf.pdf
│   ├── confusion_matrix_rf_smote.pdf
│   ├── confusion_matrix_xgb.pdf
│   ├── confusion_matrix_xgb_smote.pdf
│   ├── confusion_matrix_cat.pdf
│   ├── confusion_matrix_cat_smote.pdf
│   ├── confusion_matrix_lgbm.pdf
│   └── confusion_matrix_lgbm_smote.pdf
├── requirements.txt
└── README.md

```

---

## Dataset

### Source

The dataset was extracted from the **Portal de Dados Abertos do Estado de Minas Gerais** (Open Data Portal of the State of Minas Gerais), published on April 22, 2024. It covers tuberculosis notification records in Minas Gerais from **January 1, 2010 to April 4, 2024**.

- **Portal:** [dataviva.info](https://dataviva.info/)
- **Original system:** SINAN — *Sistema de Informação de Agravos de Notificação* (Notifiable Diseases Information System), Brazilian Ministry of Health
- **All records are anonymized** in compliance with ethical and data protection standards.

After validation and exclusion of inconsistent records, the final dataset comprises **53,656 samples** and **19 variables** (5 numerical, 14 categorical), with 1,595 missing values concentrated in sociodemographic fields.

---

### Data Dictionary

The following table describes **all columns present in the raw dataset**. Fields used in the study are marked with ✅.

| Column | Type | Label (PT) | Description (EN) | Used in Study |
|---|---|---|---|---|
| `id_agravo` | text | ID do Agravo | Unique identifier code for the notifiable disease/condition (CID-10 code for tuberculosis). | ❌ |
| `dt_notific` | text | Data de Notificação | Date on which the tuberculosis case was officially notified to the health system (DD/MM/YYYY). | ❌ |
| `nu_ano` | text | Ano de Notificação | Year of notification. Derived from `dt_notific`; used in temporal analyses. Records from 2024 excluded due to incomplete data availability. | ❌ (derived use) |
| `id_municip` | text | Município de Notificação | IBGE code of the municipality where the case was notified. High-cardinality variable; frequency encoded. | ✅ (freq. encoded) |
| `id_regiona` | text | Regional de Notificação | Regional health district code responsible for the notification. | ✅ (one-hot encoded) |
| `id_unidade` | text | Unidade de Notificação | Code for the health facility that notified the case. High-cardinality variable; frequency encoded. | ✅ (freq. encoded) |
| `dt_diag` | text | Data de Diagnóstico | Date on which the tuberculosis diagnosis was confirmed (DD/MM/YYYY). Parsed with `pd.to_datetime(format="%d/%m/%Y")`. Used to derive `tempo_inicio_trat`. | ✅ (derived) |
| `nu_idade_n` | text | Idade | Patient's age at the time of notification, in years. Used directly as a numerical feature after standardization. | ✅ (numerical) |
| `cs_sexo` | text | Sexo | Biological sex of the patient. Values: `M` (Male), `F` (Female), `I` (Ignored). | ✅ (one-hot encoded) |
| `cs_gestant` | text | Gestante | Pregnancy status at notification. Values: `1`–`3` (trimester), `4` (ignored gestational age), `5` (not applicable), `6` (not pregnant), `9` (ignored). | ❌ |
| `cs_raca` | text | Raça/Cor | Self-declared race or skin color. Values: `1` (White), `2` (Black), `3` (Yellow/Asian), `4` (Brown/Mixed), `5` (Indigenous), `9` (Ignored). | ✅ (one-hot encoded) |
| `cs_escol_n` | text | Escolaridade | Highest education level completed. Values: `0` (Illiterate), `1` (1st–4th grade incomplete), `2` (4th grade complete), `3` (5th–8th grade incomplete), `4` (Elementary complete), `5` (High school incomplete), `6` (High school complete), `7` (Higher incomplete), `8` (Higher complete), `9` (Ignored), `10` (Not applicable). | ✅ (one-hot encoded) |
| `id_mn_resi` | text | Município de Residência | IBGE code of the municipality where the patient resides. May differ from `id_municip`. High-cardinality variable; frequency encoded. | ✅ (freq. encoded) |
| `id_rg_resi` | text | Regional de Residência | Regional health district corresponding to the patient's municipality of residence. | ❌ |
| `cs_zona` | text | Zona de Residência | Residential zone. Values: `1` (Urban), `2` (Rural), `3` (Peri-urban), `9` (Ignored). | ❌ |
| `agravaids` | text | Aids | AIDS as an associated condition. After text normalization: `Sim` (Yes), `Nao` (No). | ✅ (one-hot encoded) |
| `agravalcoo` | text | Alcoolismo | Alcohol use disorder as an associated condition. Same encoding as `agravaids`. | ✅ (one-hot encoded) |
| `agravdiabe` | text | Diabetes | Diabetes as an associated condition. Same encoding as `agravaids`. | ✅ (one-hot encoded) |
| `agravdoenc` | text | Doença Mental | Mental disorder as an associated condition. Same encoding as `agravaids`. | ✅ (one-hot encoded) |
| `agravoutra` | text | Outras | Other associated conditions. Same encoding as `agravaids`. | ✅ (one-hot encoded) |
| `agravdroga` | text | Uso de Drogas Ilícitas | Illicit drug use as an associated condition. Same encoding as `agravaids`. | ❌ |
| `agravtabac` | text | Tabagismo | Tobacco use/smoking as an associated condition. Same encoding as `agravaids`. | ❌ |
| `tratamento` | text | Tipo de Entrada | Type of treatment entry. Values: `1` (New case), `2` (Relapse), `3` (Return after abandonment), `4` (Transfer), `5` (Post-cure relapse), `6` (Not informed). | ✅ (one-hot encoded) |
| `cultura_es` | text | Cultura de Escarro | Sputum (or other material) culture result for TB bacilli. Values: `1` (Positive), `2` (Negative), `3` (Contaminated), `4` (In progress), `5` (Not performed), `9` (Ignored). | ✅ (one-hot encoded) |
| `hiv` | text | HIV | HIV serology result. After text normalization: `Positiva` (Positive), `Negativa` (Negative). | ✅ (one-hot encoded) |
| `histopatol` | text | Histopatologia | Histopathological examination result for TB diagnosis. Values: `1` (Compatible/Suggestive), `2` (Not compatible), `3` (Inconclusive), `4` (Not performed), `9` (Ignored). | ✅ (one-hot encoded) |
| `dt_inic_tr` | text | Data de Início do Tratamento | Date on which the patient started the current treatment (DD/MM/YYYY). Parsed with `pd.to_datetime(format="%d/%m/%Y")`. Used to derive `tempo_inicio_trat`. | ✅ (derived) |
| `bacilosc_1` | text | Baciloscopia 1º Mês | Sputum smear microscopy at month 1. Values: `1` (Negative), `2` (+), `3` (++), `4` (+++), `5` (Not performed), `6` (Not applicable), `9` (Ignored). | ❌ |
| `bacilosc_2` | text | Baciloscopia 2º Mês | Sputum smear at month 2. Same encoding as `bacilosc_1`. | ❌ |
| `bacilosc_3` | text | Baciloscopia 3º Mês | Sputum smear at month 3. Same encoding as `bacilosc_1`. | ❌ |
| `bacilosc_4` | text | Baciloscopia 4º Mês | Sputum smear at month 4. Same encoding as `bacilosc_1`. | ❌ |
| `bacilosc_5` | text | Baciloscopia 5º Mês | Sputum smear at month 5. Same encoding as `bacilosc_1`. | ❌ |
| `bacilosc_6` | text | Baciloscopia 6º Mês | Sputum smear at month 6. Same encoding as `bacilosc_1`. | ❌ |
| `tratsup_at` | text | TDO | Directly Observed Therapy (DOT) status. Values: `1` (Daily), `2` (Weekly), `3` (Not performed), `4` (Other), `9` (Ignored). | ❌ |
| `situa_ence` | text | Situação de Encerramento | **Target variable.** Case outcome at closure. See [Target Variable](#target-variable). | ✅ (target) |
| `dt_encerra` | text | Data de Encerramento | Date on which the case was officially closed (DD/MM/YYYY). | ❌ |
| `pop_liber` | text | População Privada de Liberdade | Whether the patient belongs to the incarcerated population. Values: `1` (Yes), `2` (No), `9` (Ignored). | ❌ |
| `test_molec` | text | Teste Molecular Rápido (TMR-TB) | Rapid molecular test result (Xpert MTB/RIF or equivalent). Values: `1` (Detected — Rifampicin resistant), `2` (Detected — sensitive), `3` (Detected — indeterminate), `4` (Not detected), `5` (Inconclusive), `6` (Not performed), `9` (Ignored). | ❌ |
| `test_sensi` | text | Teste de Sensibilidade | Drug susceptibility test result. Values: `1` (Sensitive to all), `2` (Isoniazid resistant), `3` (Rifampicin resistant), `4` (MDR), `5` (Other resistance), `6` (Not performed), `9` (Ignored). | ❌ |
| `raiox_tora` | text | Radiografia do Tórax | Chest X-ray result. Values: `1` (Normal), `2` (Pleural effusion), `3` (Primary lesion), `4` (Infiltrate, no cavitation), `5` (Infiltrate with cavitation), `6` (Miliary), `7` (Other), `8` (Not performed), `9` (Ignored). | ❌ |
| `forma` | text | Forma Clínica | Clinical form of tuberculosis. Values: `1` (Pulmonary), `2` (Extrapulmonary), `3` (Pulmonary + Extrapulmonary). | ✅ (one-hot encoded) |

---

### Variables Used in the Study

The 19 selected model features plus one engineered variable:

| Variable | Encoding | Notes |
|---|---|---|
| `id_municip` | Frequency → `ID_MUNICIP_FREQ` | Relative frequency in training set |
| `id_regiona` | One-hot | |
| `id_unidade` | Frequency → `ID_UNIDADE_FREQ` | Relative frequency in training set |
| `nu_idade_n` | StandardScaler | Numerical |
| `cs_sexo` | One-hot | |
| `cs_raca` | One-hot | |
| `cs_escol_n` | One-hot | |
| `id_mn_resi` | Frequency → `ID_MN_RESI_FREQ` | Relative frequency in training set |
| `tratamento` | One-hot | |
| `cultura_es` | One-hot | |
| `histopatol` | One-hot | |
| `forma` | One-hot | |
| `agravaids` | One-hot | |
| `agravalcoo` | One-hot | |
| `agravdiabe` | One-hot | |
| `agravdoenc` | One-hot | |
| `agravoutra` | One-hot | |
| `hiv` | One-hot | |
| `tempo_inicio_trat` | StandardScaler | **Derived:** `(dt_inic_tr − dt_diag).days`; NaN imputed with median |

---

### Target Variable

Records with closure codes outside the four study classes were excluded before modeling. Internal string values after text normalization:

| Class Label | Normalized String | Count (Train) | Proportion |
|---|---|---|---|
| Cure | `Cura` | 31,663 | 73.77% |
| Abandonment | `Abandono` | 6,262 | 14.59% |
| Death from Other Causes | `Obito_por_outras_causas` | 2,810 | 6.55% |
| Death from Tuberculosis | `Obito_por_TB` | 2,189 | 5.10% |

> XGBoost, CatBoost, and LightGBM require integer targets. `LabelEncoder` is applied to the string labels before training these models.

---

## Notebooks

### `data_analysis.ipynb`

Performs all exploratory data analysis described in Section III-A of the paper. Loads `dados_tuberculose_mg.xlsx` and produces PDF visualizations. The analyses covered are:

- **Comorbidity distribution** — Percentage of all TB cases presenting each comorbidity (AIDS, alcohol use, diabetes, mental illness, HIV), split by with/without. Filters each column to `Sim`/`Não` or `Positiva`/`Negativa`, dropping ignored responses.
- **Deaths by comorbidity profile** — Among patients with outcome `Óbito por TB`, the proportional breakdown by comorbidity group.
- **Cure rate by comorbidity** — Among patients with outcome `Cura`, the proportional breakdown by comorbidity group.
- **Annual notification trend (2010–2023)** — Bar chart of absolute year-on-year change in reported cases with a twin-axis line for percentage variation. 2024 records excluded due to incomplete data.
- **Distribution by education level, sex, clinical form, and age group** — Frequency distributions for sociodemographic and clinical characterization of the reported population.


---

### `tuberculosis_outcome_classification.ipynb`

Contains the full machine learning pipeline described in Sections III-B through IV of the paper. The notebook is organized into the following sections:

**1. Imports and helper functions**  
Loads all libraries. Defines `print_metrics()` — a helper that calls `classification_report` and prints the confusion matrix for a given model and prediction set.

**2. Preprocessing**  
Applies text normalization via `unidecode`, filters the four valid outcome classes, parses date columns, derives `tempo_inicio_trat`, removes duplicates, imputes missing values with the median, applies frequency encoding to high-cardinality columns, and performs the 80/20 stratified split (`random_state=43`). Builds the `ColumnTransformer` with `OneHotEncoder` for categoricals and `StandardScaler` for numericals.

**3. Model training and cross-validation (all four models)**  
For each model, trains two pipelines — one using `sklearn.pipeline.Pipeline` (no SMOTE) and one using `imblearn.pipeline.Pipeline` (with SMOTE inserted between preprocessing and the classifier). Evaluates each with `cross_val_score` (5-fold stratified CV, `f1_weighted`). Reports per-fold scores, mean, and standard deviation. XGBoost, CatBoost, and LightGBM use `LabelEncoder` to convert string labels to integers.

**4. Test set evaluation and confusion matrices**  
Refits each pipeline on the full training set and evaluates on the held-out test set. Generates confusion matrices using `ConfusionMatrixDisplay` with a custom yellow-blue colormap (`LinearSegmentedColormap`). Exports each matrix as a PDF (e.g., `confusion_matrix_rf.pdf`, `confusion_matrix_cat_smote.pdf`).

**5. Full experiment report**  
Consolidated summary of dataset shape, class distributions for train/test and SMOTE-balanced sets, and cross-validation statistics (mean, std, 95% CI) across all 8 model-SMOTE configurations. Also runs multi-seed stability experiments (5 seeds) for the ablation study and feature-subset analysis, using Random Forest as the fixed baseline throughout.

**6. Temporal (walk-forward) validation**  
For each year `t` from the second available year onward, trains on all records with `YEAR < t` and tests on records with `YEAR == t`. Years with fewer than 30 test records are skipped. SMOTE `k_neighbors` is dynamically set to `min(5, min_class_size − 1)` to handle small class sizes in early training windows. Exports weighted F1-score and per-class recall across test years as `temporal_forward_validation.pdf`.

**7. Class distribution visualization**  
Generates the grouped bar chart (using `seaborn.barplot`) comparing class proportions in the original training set, test set, and SMOTE-balanced training set. Exported as PDF.

---

## Methodology

### Preprocessing Pipeline

1. **Text normalization:** Accents removed via `unidecode`, whitespace trimmed, spaces replaced with underscores — applied to all `object`-type columns.
2. **Outcome filtering:** Only `Cura`, `Abandono`, `Obito_por_TB`, `Obito_por_outras_causas` retained.
3. **Date parsing:** `dt_diag` and `dt_inic_tr` converted with `pd.to_datetime(format="%d/%m/%Y", errors="coerce")`. `tempo_inicio_trat` derived as `(dt_inic_tr − dt_diag).days`. Missing values imputed with the **median**.
4. **Deduplication:** `drop_duplicates()` applied before splitting.
5. **Frequency encoding:** `ID_MUNICIP`, `ID_UNIDADE`, `ID_MN_RESI` replaced by their relative frequencies computed on the preprocessed dataset.
6. **Train/test split:** 80/20 stratified hold-out (`random_state=43`) → 42,924 train / 10,732 test samples.
7. **ColumnTransformer:**
   - Low-cardinality categoricals → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
   - Numerical features → `StandardScaler()`
8. **Pipeline / ImbPipeline:** All transformations fitted exclusively on training data. SMOTE inserted between preprocessing and the classifier in `ImbPipeline`, preventing leakage into validation or test sets.

### Class Imbalance Handling

SMOTE (`k_neighbors=5`) applied only within training folds — never on the test set. After SMOTE, all four classes are equally represented with **31,663 samples each** (25.00%), totaling 126,652 training samples. In the temporal walk-forward experiment, `k_neighbors` is dynamically reduced to `min(5, min_class_size − 1)` when small classes appear in early training windows.

### Model Configuration

| Model | Key Hyperparameters |
|---|---|
| Random Forest | `n_estimators=200`, `class_weight="balanced"`, `random_state=43` |
| XGBoost | `n_estimators=300`, `learning_rate=0.1`, `max_depth=6`, `eval_metric="mlogloss"`, `random_state=43` |
| CatBoost | `iterations=300`, `learning_rate=0.1`, `depth=6`, `loss_function="MultiClass"`, `verbose=0`, `random_state=43` |
| LightGBM | `n_estimators=300`, `learning_rate=0.1`, `max_depth=-1`, `objective="multiclass"`, `class_weight="balanced"`, `random_state=43` |

Hyperparameters were selected through manual empirical tuning.

### Evaluation Strategy

- **Primary metrics:** Accuracy, per-class precision/recall/F1, weighted F1-score
- **Cross-validation:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=43)` on the training set
- **Multi-seed experiments:** 5 random seeds for stability assessment in ablation and feature-subset analyses
- **Ablation study:** Baseline → + Feature Engineering → + SMOTE, using Random Forest as fixed baseline
- **Feature subset analysis:** Clinical only vs. socioeconomic only vs. complete feature set
- **Temporal validation:** Walk-forward across all available years, SMOTE applied per-fold with dynamic `k_neighbors`

---

## Results Summary

| Model | Setting | CV F1 (mean ± std) | Test Accuracy |
|---|---|---|---|
| Random Forest | No SMOTE | 0.6780 ± 0.0028 | 0.7461 |
| Random Forest | SMOTE | 0.6930 ± 0.0039 | 0.7229 |
| XGBoost | No SMOTE | 0.6976 ± 0.0027 | 0.7486 |
| XGBoost | SMOTE | 0.7048 ± 0.0029 | 0.7395 |
| CatBoost | No SMOTE | 0.6933 ± 0.0042 | 0.7522 |
| **CatBoost** | **SMOTE** | **0.7071 ± 0.0031** | **0.7301** |
| LightGBM | No SMOTE | 0.6353 ± 0.0016 | 0.5905 |
| LightGBM | SMOTE | 0.7025 ± 0.0012 | 0.7432 |

**CatBoost + SMOTE** achieved the best cross-validation F1-score and highest minority-class recall: Abandonment (0.27), Death from TB (0.15), Death from Other Causes (0.23).

---

## Requirements

```
python==3.12.4
pandas==2.2.2
numpy==1.26.4
unidecode==1.2.0
scikit-learn==1.4.2
imbalanced-learn==0.12.3
xgboost==3.0.5
catboost==1.2.8
lightgbm==4.6.0
matplotlib==3.8.4
seaborn==0.13.2
scipy==1.13.1
folium==0.20.0
openpyxl==3.1.2
jupyter

```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/tuberculosis-treatment-outcomes-ml.git
   cd tuberculosis-treatment-outcomes-ml
   ```

2. **Download the dataset** from the [Minas Gerais Open Data Portal](https://dataviva.info/) and place the file at:
   ```
   data/dados_tuberculose_mg.xlsx
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebooks:**
   ```bash
   jupyter notebook
   ```
   - `data_analysis.ipynb` — run first for exploratory analysis and figures.
   - `tuberculosis_outcome_classification.ipynb` — run for the full ML pipeline.

---