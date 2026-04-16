"""
M6a — CONFIRMATORY (pre-registered). DO NOT RUN until plan is committed.

Permutation test for interaction: model x extremity tercile.
H1: gap(T3-T1) differs between HUG-trained and pex-trained on CMHA (ablation C).

Null: exchange predictions randomly between models, preserving tercile structure.
Test statistic: diff_gap = gap_HUG - gap_pex.
5000 permutations. Two-sided p < 0.05.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]

N_PERM = 5000
SEED = 42


def load_all():
    df = pd.read_csv(IN_CSV)
    hug = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["hug2016", "hug2016snf"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    pex = df[(df.cohort == "AneuX")
             & df.subcohort.isin(["aneurisk", "aneurist"])
             & df.location.fillna("").str.contains("MCA")
             & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    cmha = df[(df.cohort == "CMHA") & df.ruptured.notna()]\
              .dropna(subset=FEATURES_FULL).reset_index(drop=True)
    return hug, pex, cmha


def fit_predict_cmha(train_df, cmha, feats):
    Xt = train_df[feats].values.astype(float)
    yt = train_df["ruptured"].astype(int).values
    Xc = cmha[feats].values.astype(float)
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")
    clf.fit(sc.transform(Xt), yt)
    return clf.predict_proba(sc.transform(Xc))[:, 1]


def compute_extremity(cmha):
    unr = cmha[cmha.ruptured == 0]
    mu = unr[FEATURES_FULL].mean()
    sd = unr[FEATURES_FULL].std(ddof=1).replace(0, 1)
    z = ((cmha[FEATURES_FULL] - mu) / sd).abs().mean(axis=1)
    return z.values


def compute_gap(p_model, y, extrem, tercile_cuts):
    rom_mask = y == 1
    unr_mask = y == 0
    aucs = []
    for lo, hi in tercile_cuts:
        t_mask = rom_mask & (extrem > lo) & (extrem <= hi)
        y_sub = np.concatenate([np.ones(t_mask.sum()), np.zeros(unr_mask.sum())])
        p_sub = np.concatenate([p_model[t_mask], p_model[unr_mask]])
        if len(np.unique(y_sub)) < 2:
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(y_sub, p_sub))
    return aucs[2] - aucs[0]


def main():
    rng = np.random.default_rng(SEED)
    hug, pex, cmha = load_all()
    feats = NON_NPS

    p_hug = fit_predict_cmha(hug, cmha, feats)
    p_pex = fit_predict_cmha(pex, cmha, feats)
    y = cmha["ruptured"].astype(int).values
    extrem = compute_extremity(cmha)

    rom_ext = extrem[y == 1]
    cut1 = float(np.quantile(rom_ext, 1/3))
    cut2 = float(np.quantile(rom_ext, 2/3))
    tercile_cuts = [(-np.inf, cut1), (cut1, cut2), (cut2, np.inf)]

    gap_hug_obs = compute_gap(p_hug, y, extrem, tercile_cuts)
    gap_pex_obs = compute_gap(p_pex, y, extrem, tercile_cuts)
    diff_obs = gap_hug_obs - gap_pex_obs

    print("=" * 60)
    print("M6a — Permutation test: interaction model × extremity tercile")
    print("=" * 60)
    print(f"Observed gap HUG-trained (T3-T1): {gap_hug_obs:+.3f}")
    print(f"Observed gap pex-trained (T3-T1): {gap_pex_obs:+.3f}")
    print(f"Observed diff (gap_HUG - gap_pex): {diff_obs:+.3f}")
    print(f"\nRunning {N_PERM} permutations...")

    null_diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        swap = rng.random(len(y)) < 0.5
        p_h_perm = np.where(swap, p_pex, p_hug)
        p_p_perm = np.where(swap, p_hug, p_pex)
        g_h = compute_gap(p_h_perm, y, extrem, tercile_cuts)
        g_p = compute_gap(p_p_perm, y, extrem, tercile_cuts)
        null_diffs[i] = g_h - g_p

    p_two_sided = np.mean(np.abs(null_diffs) >= np.abs(diff_obs))

    print(f"\nNull distribution: mean={null_diffs.mean():.3f}, "
          f"sd={null_diffs.std():.3f}")
    print(f"Observed diff: {diff_obs:+.3f}")
    print(f"Two-sided p-value: {p_two_sided:.4f}")
    print()

    if p_two_sided < 0.05:
        print(">>> SIGNIFICANT (p < 0.05): interaction model × tercile confirmed.")
    else:
        print(">>> NOT significant (p >= 0.05): interaction not confirmed.")

    results = pd.DataFrame([{
        "test": "permutation_interaction",
        "gap_hug_obs": gap_hug_obs,
        "gap_pex_obs": gap_pex_obs,
        "diff_obs": diff_obs,
        "null_mean": null_diffs.mean(),
        "null_sd": null_diffs.std(),
        "p_two_sided": p_two_sided,
        "n_perm": N_PERM,
        "significant_005": p_two_sided < 0.05,
    }])
    results.to_csv(OUT_DIR / "M6a_permutation_interaction.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M6a_permutation_interaction.csv'}")


if __name__ == "__main__":
    main()
