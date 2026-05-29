# healthcare_provider_claims_fraud
An end-to-end machine learning pipeline for identifying potentially fraudulent healthcare providers using Medicare claims data. The project takes raw beneficiary, inpatient, and outpatient claims CSVs, engineers provider-level behavioral features, trains and compares multiple classifiers under severe class imbalance (~9% fraud), and produces per-provider explainability reports suitable for investigation teams.

**Dataset:** [Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/data) (Kaggle)

---

## Table of Contents

- [Problem Context](#problem-context)
- [Dataset Structure](#dataset-structure)
- [Pipeline Overview](#pipeline-overview)
  - [1. Data Loading & Exploration](#1-data-loading--exploration)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Exploratory Analysis](#3-exploratory-analysis)
  - [4. Class Imbalance Strategies](#4-class-imbalance-strategies)
  - [5. Model Training & Selection](#5-model-training--selection)
  - [6. Explainability with SHAP](#6-explainability-with-shap)
  - [7. Evaluation & Operational Thresholds](#7-evaluation--operational-thresholds)
- [Key Technical Decisions](#key-technical-decisions)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [Future Work](#future-work)

---

## Problem Context

Medicare fraud costs the US healthcare system tens of billions of dollars annually. Special Investigation Units (SIUs) at health plans have limited bandwidth to review providers, so they need a scoring system that ranks providers by fraud likelihood and explains *why* each was flagged. This project simulates that workflow: ingest claims data, build a provider risk profile, score, rank, and explain.

The dataset contains labeled providers (~9% flagged as potentially fraudulent), making this a supervised binary classification problem with significant class imbalance.

---

## Dataset Structure

The raw data consists of four CSVs that must be joined and aggregated:

| File | Grain | Key Fields |
|------|-------|------------|
| `Train.csv` | Provider | Provider ID, PotentialFraud label (Yes/No) |
| `Train_Beneficiary.csv` | Patient | Demographics, chronic conditions (11 flags + RenalDiseaseIndicator), annual reimbursement/deductible amounts, DOB/DOD |
| `Train_Inpatient.csv` | Claim | Diagnosis codes (1–10), procedure codes (1–6), reimbursement, deductible, attending/operating/other physicians, admission/discharge dates |
| `Train_Outpatient.csv` | Claim | Similar to inpatient but without admission/discharge dates |

The modeling target is at the **provider level**, so the core challenge is aggregating claim-level and patient-level data into meaningful provider-level features.

---

## Pipeline Overview

### 1. Data Loading & Exploration

- Load all four CSVs with explicit dtype handling (e.g., `RenalDiseaseIndicator` is mixed-type and must be read as `str` to preserve the `'Y'`/`'0'` encoding)
- Profile missingness across all tables — operating/other physician fields are sparse in outpatient (expected), diagnosis codes 5–10 are sparse (signals complexity, not missing data)
- Validate target distribution: ~9% fraud prevalence

### 2. Feature Engineering

This is the core of the project. Raw claims have no predictive power — the signal lives entirely in provider-level aggregations.

**Date & Duration Features**
- Claim duration (end − start), length of stay (discharge − admission), patient age at claim time

**Clinical Complexity**
- Count of non-null diagnosis codes per claim (both inpatient and outpatient)
- Count of non-null procedure codes per claim (both inpatient and outpatient — outpatient procedure density is itself a fraud signal indicating potential upcoding)
- Chronic condition burden per patient, including the `RenalDiseaseIndicator` (ESRD), which uses a different encoding scheme (`'Y'`/`'0'`) than the `ChronicCond_*` columns (`1`/`2`) and must be cleaned separately before aggregation

**Unified Claims Table**
- Inpatient and outpatient claims are tagged by type and concatenated with shared columns including `OperatingPhysician` and `OtherPhysician` (retained for physician network analysis despite sparsity in outpatient — their presence on outpatient claims is itself anomalous and fraud-relevant)
- Beneficiary features merged in: chronic count, ESRD flag, deceased flag, annual reimbursement amounts, and annual deductible amounts (`IPAnnualDeductibleAmt`, `OPAnnualDeductibleAmt`)

**Provider-Level Aggregation** 
- Volume: total claims, unique patients, unique physicians
- Financial: mean/median/std/max reimbursement, mean deductible, total reimbursement
- Duration: mean claim duration, mean length of stay
- Complexity: mean/max diagnosis codes, mean procedure codes
- Patient panel: mean age, mean chronic count, percent deceased patients
- Mix: percent inpatient claims
- Annual financials: mean IP/OP annual reimbursement, mean IP/OP annual deductible

**Derived Ratios**
- Claims per patient (churning signal)
- Reimbursement per patient
- Coefficient of variation of reimbursement (billing pattern consistency)
- Deductible exhaustion rate (reimbursement ÷ deductible — flags providers targeting patients with maxed-out benefits)
- Reimbursement-to-deductible ratio
- Physician concentration (Herfindahl index — flags providers where one physician dominates billing)

### 3. Exploratory Analysis

- KDE distribution comparison of every feature, split by fraud label
- Correlation analysis of all features against the target
- SHAP beeswarm plot (post-modeling) used to validate or override domain intuition about which features actually matter

### 4. Class Imbalance Strategies

Four approaches compared on equal footing:

| Strategy | Implementation |
|----------|---------------|
| Logistic Regression + class weights | `class_weight='balanced'` with `StandardScaler` in pipeline |
| Logistic Regression + SMOTE | `imblearn` pipeline: `StandardScaler` → `SMOTE` → `LogisticRegression` |
| Random Forest + class weights | `class_weight='balanced'`, 300 estimators |
| XGBoost + scale_pos_weight | `scale_pos_weight = count_legit / count_fraud`, tuned hyperparameters |

Both logistic regression variants are wrapped in pipelines with `StandardScaler` to ensure performance differences reflect the resampling strategy, not accidental scaling effects. Tree-based models (XGBoost, RandomForest) do not require scaling.

### 5. Model Training & Selection

- 5-fold Stratified K-Fold cross-validation to preserve fraud ratio in each fold
- Dual-metric evaluation: ROC-AUC and PR-AUC (`average_precision_score`), with PR-AUC as the primary selection criterion because ROC-AUC is misleading under class imbalance
- Hyperparameter tuning via `RandomizedSearchCV` (50 iterations) for the best-performing model family
- Search space covers: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`

### 6. Explainability with SHAP

Explainability is non-negotiable in healthcare fraud — investigations go to regulators and sometimes courts.

**Model-agnostic explainer selection:**
- Tree-based models (XGBoost, RandomForest) → `shap.TreeExplainer`
- Linear models (LogisticRegression) → `shap.LinearExplainer`
- Pipeline models (SMOTE + LR) → extract the final estimator and replay preprocessing; use resampled data as SHAP background but compute values on real providers only

**Global explanations:**
- SHAP beeswarm plot showing feature importance + direction of effect across all providers
- Feature importance bar chart

**Per-provider explanations:**
- `explain_provider()` function generates a human-readable report for any provider: fraud probability, top 5 risk drivers with actual values vs. population averages, and directional indicators (↑ RISK / ↓ risk)
- Designed for investigator consumption: "This provider was flagged because their average reimbursement is 3× the population mean and 40% of their patient panel is deceased"

### 7. Evaluation & Operational Thresholds

- Standard metrics: ROC-AUC, PR-AUC, classification report
- Operationalized threshold selection: rather than using 0.5, the threshold is set based on investigator capacity (e.g., "flag the top 15% of providers for review") using `np.percentile`
- Threshold sweep across 5%/10%/15%/20%/25% flag rates, showing precision-recall tradeoffs at each level so the investigations team can choose based on their bandwidth
- Confusion matrix visualization

---

## Key Technical Decisions

**Why PR-AUC over ROC-AUC?** With ~9% fraud prevalence, a model predicting "not fraud" for everything achieves 91% accuracy and a deceptively reasonable ROC-AUC. PR-AUC directly measures "of the providers we flag, how many are actually fraudulent?" — the metric that maps to investigator productivity.

**Why include sparse columns?** Columns like `OperatingPhysician` (mostly null in outpatient) and outpatient procedure codes were retained rather than dropped because their *presence* in unexpected contexts is itself a fraud signal. A provider billing operating physician involvement on routine outpatient visits is anomalous.

**Why explicit dtype handling?** The `RenalDiseaseIndicator` column contains mixed types (`'Y'` and `'0'`). On a clean first read, pandas infers `object` dtype. If the column is ever cleaned and re-saved, subsequent reads infer `int64`, silently changing behavior. Pinning `dtype={'RenalDiseaseIndicator': str}` at read time makes the pipeline deterministic.

**Why scale LR but not XGBoost?** Logistic regression learns additive coefficients — features on different scales get unequal regularization. Tree-based models split on rank order, making them scale-invariant. Both LR variants are wrapped in `StandardScaler` pipelines so their comparison isolates the effect of SMOTE vs. class weights.

---

## Results


| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| LR + class weights | — | — |
| LR + SMOTE | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

Top features by SHAP importance: `total_reimbursement`, `claims_per_patient`, `mean_los`, `max_diag_codes`, `max_reimbursement`, `total_claims`.

---
