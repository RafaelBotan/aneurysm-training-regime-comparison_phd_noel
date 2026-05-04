# Aneurysm Training Regime Comparison

**Severe External Miscalibration Despite High Internal Discrimination in Aneurysm Rupture Models**

OSF reproducibility package and DOI: <https://doi.org/10.17605/OSF.IO/S4YVB>

GitHub repository: <https://github.com/RafaelBotan/aneurysm-training-regime-comparison_phd_noel>

## Overview

This repository contains analysis code for a study comparing two training regimes for morphology-only intracranial aneurysm rupture prediction:

- **HUG-trained** (n=118, 14 ruptured) — larger cohort with morphologically extreme positive class
- **pex-trained** (n=70, 30 ruptured) — smaller cohort with more representative rupture phenotype

Both regimes are evaluated across internal cross-validation, cross-cohort external validation, and an out-of-domain stress cohort (CMHA, n=105, 77 ruptured).

The study shows that high internal discrimination can coexist with severe external miscalibration after dataset shift. The overall pattern is consistent with positive-class spectrum bias, while the pre-specified regime-by-extremity permutation test was directionally positive but did not meet the confirmatory threshold.

## Repository structure

```
00_briefing/          Analysis plan and locked follow-up plan
02_codigo/            Analysis scripts (M2-M6e)
03_analises/          Selected outputs and figures
```

### Analysis pipeline

| Script | Description |
|--------|-------------|
| `M2_shared_feature_core.py` | Harmonization of AneuX + CMHA into shared 11-feature core |
| `M3_hug_mca_to_cmha.py` | Baseline: HUG MCA → CMHA transport |
| `M4_cascade_external.py` | Cascade external validation (AneuX → CMHA) |
| `M4b_bootstrap_ci.py` | Bootstrap 95% CI on cascade |
| `M4c_site_shift_audit.py` | KS + Cohen's d site-shift audit within AneuX |
| `M4d_ruptured_distribution.py` | Within-cohort ruptured vs unruptured separation |
| `M4e_reverse_train_pseudoext.py` | Reverse-train: pex → HUG + CMHA |
| `M4f_bootstrap_m4e.py` | Bootstrap CI on reverse-train results |
| `M4g_extremes_overlap.py` | Mechanism: extremes-overlap by morphological tercile |
| `M4h_leave_block_out_extremity.py` | Out-of-model extremity (circularity blindage) |
| `M4i_bootstrap_gap_tercil.py` | Bootstrap interaction: model × extremity tercile |
| `M5_figures.py` | Main figures (Fig 1-4) |
| `M5b_table1.py` | Table 1: 3-cohort descriptive statistics |
| `M5c_sensitivity.py` | Sensitivity analyses (mixed patients, pooled) |
| `M5d_factorial_2x2.py` | Factorial 2x2: regime × ablation |
| `M5e_hug1_only.py` | HUG1-only sensitivity (spectrum bias source) |
| `M6a_permutation_interaction.py` | Confirmatory permutation test for regime-by-extremity interaction |
| `M6b_bootstrap_factorial_grid.py` | Bootstrap grid for factorial regime/ablation effects |
| `M6c_delong_test.py` | DeLong comparison on CMHA |
| `M6d_calibration_formal.py` | Formal calibration metrics |
| `M6e_leave_k_out_stability.py` | Leave-5-out stability analysis |

### Key figures

- **Fig 1**: Study flowchart
- **Fig 2**: ROC curves — 2 regimes × 3 datasets
- **Fig 3**: Calibration plots — 2 regimes × 3 datasets
- **Fig 4**: Cohen's d heatmap — within-cohort separation
- **Fig 5**: AUC by morphological extremity tercile

## Data availability

Raw morphological measurements are not included in this repository. AneuX data is available from <https://aneux.ch>. CMHA data is available from figshare at <https://doi.org/10.6084/m9.figshare.26965450>. The reproducibility package for the manuscript is archived on OSF at <https://doi.org/10.17605/OSF.IO/S4YVB>.

## Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn, matplotlib
