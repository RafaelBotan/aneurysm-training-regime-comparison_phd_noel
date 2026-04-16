# Confirmatory Analysis Plan — Pre-registered after Exploratory Phase

**Date:** 2026-04-15
**Status:** LOCKED (committed before execution)
**Repository:** github.com/RafaelBotan/aneurysm-training-regime-comparison_phd_noel

## Context

The exploratory phase (scripts M3–M5e) identified positive-class spectrum bias
as a candidate mechanism explaining the high internal AUC (0.88) of the
HUG-trained model. The exploratory findings are summarized in Memorial v6.1
(`00_briefing/M1_memorial_v6.md`).

This confirmatory plan specifies five pre-registered analyses with exact
hypotheses, test statistics, and decision criteria. These analyses have NOT
been executed at the time of this registration. The scripts exist in the
repository but have not been run.

## Exploratory findings (declared post-hoc)

- M3–M5e: full pipeline from harmonization to sensitivity analyses
- Key finding: HUG-trained model shows AUC gradient by morphological
  extremity tercile (T1=0.61, T2=0.51, T3=0.84), pex-trained does not
- Factorial 2×2 grid computed (4 cells × 3 datasets)
- Bootstrap interaction CI excludes zero (exploratory bootstrap)

## Confirmatory Hypotheses

### H1 — Permutation test for interaction (model × extremity tercile)

**Hypothesis:** The gap (AUC_T3 − AUC_T1) differs significantly between
HUG-trained and pex-trained models on CMHA, using ablation C (6 non-NPS
features).

**Test:** Two-sided permutation test (5000 permutations). Under the null,
CMHA ruptured case predictions from HUG-trained and pex-trained are exchanged
at random, preserving tercile assignments. Test statistic = difference of gaps
(gap_HUG − gap_pex).

**Decision:** Significant if two-sided p < 0.05.

**Script:** `02_codigo/M6a_permutation_interaction.py`

### H2 — Bootstrap CI for full factorial 2×2 grid

**Hypothesis:** Each cell of the 2×2 grid (regime × ablation) has a well-defined
AUC with quantifiable uncertainty on each of the 3 datasets (12 total CIs).

**Test:** Stratified bootstrap (2000 resamples, stratified by outcome) for each
of the 12 AUC estimates (4 cells × 3 datasets).

**Report:** Point estimate + 95% percentile CI for all 12 cells. No decision
threshold — descriptive.

**Script:** `02_codigo/M6b_bootstrap_factorial_grid.py`

### H3 — DeLong test for regime difference on CMHA

**Hypothesis:** On CMHA (ablation C), pex-trained AUC > HUG-trained AUC.

**Test:** DeLong test (one-sided) comparing AUC of pex-trained vs HUG-trained
on CMHA, ablation C. Both models applied to the same CMHA cases.

**Decision:** Significant if one-sided p < 0.05.

**Script:** `02_codigo/M6c_delong_test.py`

### H4 — Formal calibration metrics with CI

**Hypothesis:** HUG-trained models show worse calibration than pex-trained
on external datasets.

**Test:** Bootstrap (1000 resamples) for Brier score, calibration slope, and
calibration intercept, per cell of the 2×2 grid on each external dataset
(CMHA + cross-external). Total: 4 cells × 2 external datasets × 3 metrics = 24.

**Report:** Point estimate + 95% CI. Decision: HUG-trained calibration is
"worse" if its Brier score CI is entirely above pex-trained Brier score CI
on CMHA.

**Script:** `02_codigo/M6d_calibration_formal.py`

### H5 — Leave-k-out stability of tercile gap

**Hypothesis:** The interaction (gap_HUG − gap_pex) is stable under
perturbation of CMHA ruptured cases.

**Test:** Leave-5-out jackknife on CMHA ruptured cases (77 total, 500 random
draws of 5 to exclude). For each draw, recompute tercile assignments and
gaps for both models.

**Report:** Distribution of interaction values. Fraction of draws where
interaction > 0. Decision: stable if >90% of draws show interaction > 0.

**Script:** `02_codigo/M6e_leave_k_out_stability.py`

## Data

All analyses use `03_analises/M2_shared_feature_core.csv` (unchanged from
exploratory phase). No new data collection.

## Features

- **Ablation C (primary):** EI, NSI, UI, Dmax, S, V (6 non-NPS features)
- **Full-11 (secondary):** AR, BF, CP, EI, NSI, UI, Dmax, Dn, H, S, V

## Model specification

Logistic regression, L2 penalty, C=1.0, solver=lbfgs, max_iter=5000,
class_weight='balanced', StandardScaler fit on training data only.
Random seed=42 for reproducibility.

## Cohorts

- **HUG MCA:** n=118, 14 ruptured (training regime 1)
- **Pseudo-external MCA:** n=70, 30 ruptured (training regime 2)
- **CMHA MCA:** n=105, 77 ruptured (out-of-domain stress cohort)

## Transparency note

This is NOT a classical pre-registration. The exploratory phase was conducted
first, and the confirmatory hypotheses were derived from exploratory findings.
We register this plan to:
1. Commit to specific statistical tests before seeing their results
2. Prevent post-hoc selection of confirmatory tests
3. Provide a verifiable timestamp via git commit hash
