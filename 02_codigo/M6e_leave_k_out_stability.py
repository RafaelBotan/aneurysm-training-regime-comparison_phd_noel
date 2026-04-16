"""
M6e — CONFIRMATORY (pre-registered). DO NOT RUN until plan is committed.

Leave-k-out stability of the interaction (gap_HUG - gap_pex).
Leave 5 CMHA ruptured cases out at a time (500 random draws).
Recompute tercile assignments and gaps for both models.
Stable if >90% of draws show interaction > 0.
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

K_OUT = 5
N_DRAWS = 500
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


def fit_predict_cmha(train_df, cmha_df, feats):
    Xt = train_df[feats].values.astype(float)
    yt = train_df["ruptured"].astype(int).values
    Xc = cmha_df[feats].values.astype(float)
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")
    clf.fit(sc.transform(Xt), yt)
    return clf.predict_proba(sc.transform(Xc))[:, 1]


def compute_interaction(p_hug, p_pex, y, extrem):
    rom_mask = y == 1
    unr_mask = y == 0
    rom_ext = extrem[rom_mask]

    if len(rom_ext) < 6:
        return np.nan

    cut1 = float(np.quantile(rom_ext, 1/3))
    cut2 = float(np.quantile(rom_ext, 2/3))
    tercile_cuts = [(-np.inf, cut1), (cut1, cut2), (cut2, np.inf)]

    def gap(p_model):
        aucs = []
        for lo, hi in tercile_cuts:
            t_mask = rom_mask & (extrem > lo) & (extrem <= hi)
            if t_mask.sum() < 2:
                aucs.append(0.5)
                continue
            y_sub = np.concatenate([np.ones(t_mask.sum()), np.zeros(unr_mask.sum())])
            p_sub = np.concatenate([p_model[t_mask], p_model[unr_mask]])
            if len(np.unique(y_sub)) < 2:
                aucs.append(0.5)
            else:
                aucs.append(roc_auc_score(y_sub, p_sub))
        return aucs[2] - aucs[0]

    return gap(p_hug) - gap(p_pex)


def main():
    rng = np.random.default_rng(SEED)
    hug, pex, cmha = load_all()
    feats = NON_NPS

    p_hug_full = fit_predict_cmha(hug, cmha, feats)
    p_pex_full = fit_predict_cmha(pex, cmha, feats)
    y_full = cmha["ruptured"].astype(int).values

    unr_cmha = cmha[cmha.ruptured == 0]
    mu = unr_cmha[FEATURES_FULL].mean()
    sd = unr_cmha[FEATURES_FULL].std(ddof=1).replace(0, 1)
    extrem_full = ((cmha[FEATURES_FULL] - mu) / sd).abs().mean(axis=1).values

    interaction_full = compute_interaction(p_hug_full, p_pex_full, y_full, extrem_full)

    print("=" * 60)
    print("M6e — Leave-k-out stability of interaction (k=5, 500 draws)")
    print("=" * 60)
    print(f"Full-sample interaction: {interaction_full:+.3f}")
    print(f"\nRunning {N_DRAWS} draws (removing {K_OUT} ruptured each)...")

    rom_idx = np.where(y_full == 1)[0]
    interactions = np.empty(N_DRAWS)

    for i in range(N_DRAWS):
        exclude = rng.choice(rom_idx, K_OUT, replace=False)
        keep_mask = np.ones(len(cmha), dtype=bool)
        keep_mask[exclude] = False

        interactions[i] = compute_interaction(
            p_hug_full[keep_mask], p_pex_full[keep_mask],
            y_full[keep_mask], extrem_full[keep_mask],
        )

    frac_positive = np.mean(interactions > 0)
    mean_int = np.nanmean(interactions)
    sd_int = np.nanstd(interactions)
    ci = np.nanquantile(interactions, [0.025, 0.975])

    print(f"\nResults:")
    print(f"  Mean interaction:     {mean_int:+.3f}")
    print(f"  SD:                   {sd_int:.3f}")
    print(f"  95% range:            [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  Fraction > 0:         {frac_positive:.3f} ({frac_positive*100:.1f}%)")
    print()

    if frac_positive > 0.90:
        print(f">>> STABLE: {frac_positive*100:.1f}% > 90% threshold. "
              f"Interaction robust to dropping {K_OUT} ruptured cases.")
    else:
        print(f">>> UNSTABLE: {frac_positive*100:.1f}% <= 90% threshold. "
              f"Interaction sensitive to case composition.")

    results = pd.DataFrame([{
        "test": "leave_k_out_stability",
        "k_out": K_OUT,
        "n_draws": N_DRAWS,
        "interaction_full": interaction_full,
        "interaction_mean": mean_int,
        "interaction_sd": sd_int,
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "frac_positive": frac_positive,
        "stable_90pct": frac_positive > 0.90,
    }])
    results.to_csv(OUT_DIR / "M6e_leave_k_out_stability.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M6e_leave_k_out_stability.csv'}")


if __name__ == "__main__":
    main()
