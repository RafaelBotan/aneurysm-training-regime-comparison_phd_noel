# Aneurysm Training Regime Comparison

**Internal Discrimination Consistent with Positive-Class Spectrum Bias: A Factorial Comparison of Training Regimes and Feature Sets for Morphology-Only Rupture Models Across Harmonized and Out-of-Domain Cohorts**

## Overview

This repository contains the analysis code for a study comparing two training regimes for morphology-only intracranial aneurysm rupture prediction:

- **HUG-trained** (n=118, 14 ruptured) — larger cohort with morphologically extreme positive class
- **pex-trained** (n=70, 30 ruptured) — smaller cohort with more representative rupture phenotype

Both regimes are evaluated across internal cross-validation, cross-cohort external validation, and an out-of-domain stress cohort (CMHA, n=105, 77 ruptured).

The study demonstrates that the high internal AUC (0.88) observed in the HUG-trained model is consistent with positive-class spectrum bias rather than a generalizable rupture signal.

## Repository structure

```
00_briefing/          Analysis plan (Memorial v6.1)
02_codigo/            Analysis scripts (M2-M5e)
03_analises/figures/  Output figures (Fig 1-4)
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

### Key figures

- **Fig 1**: AUC by morphological extremity tercile (hero figure)
- **Fig 2**: ROC curves — 2 regimes × 3 datasets
- **Fig 3**: Calibration plots — 2 regimes × 3 datasets
- **Fig 4**: Cohen's d heatmap — within-cohort separation

## Data availability

Raw morphological measurements are not included in this repository. AneuX data is available from [AneuriskWeb](http://ecm2.mathcs.emory.edu/aneuriskweb/index) and individual source repositories. CMHA data access requires institutional agreement.

## Requirements

- Python 3.10+
- numpy, pandas, scipy, scikit-learn, matplotlib
