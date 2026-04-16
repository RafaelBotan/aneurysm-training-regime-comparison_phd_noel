"""
M6b — CONFIRMATORY (pre-registered). DO NOT RUN until plan is committed.

Bootstrap 95% CI for each cell of the 2x2 factorial grid (regime x ablation)
across all 3 datasets. 12 CIs total. Stratified by outcome.
2000 resamples. Descriptive (no decision threshold).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]

ABLATIONS = {"full_11": FEATURES_FULL, "sem_NPS": NON_NPS}
N_BOOT = 2000
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


def strat_boot_auc(y, p, n_boot, rng):
    y = np.asarray(y); p = np.asarray(p)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos, len(pos), replace=True)
        bn = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        aucs[i] = roc_auc_score(y[idx], p[idx])
    return aucs


def main():
    rng = np.random.default_rng(SEED)
    hug, pex, cmha = load_all()

    regimes = {
        "HUG": (hug, {"cross_ext": pex, "CMHA": cmha}),
        "pex": (pex, {"cross_ext": hug, "CMHA": cmha}),
    }

    print("=" * 76)
    print("M6b — Bootstrap 95% CI for factorial 2×2 grid (2000 resamples)")
    print("=" * 76)
    print()

    rows = []
    for regime, (train_df, test_dict) in regimes.items():
        for abl_name, feats in ABLATIONS.items():
            Xt = train_df[feats].values.astype(float)
            yt = train_df["ruptured"].astype(int).values
            sc = StandardScaler().fit(Xt)
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                     max_iter=5000, class_weight="balanced")
            clf.fit(sc.transform(Xt), yt)

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            p_cv = cross_val_predict(
                LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=5000, class_weight="balanced"),
                sc.transform(Xt), yt, cv=cv, method="predict_proba",
            )[:, 1]

            auc_cv = roc_auc_score(yt, p_cv)
            boot_cv = strat_boot_auc(yt, p_cv, N_BOOT, rng)
            ci_cv = np.quantile(boot_cv, [0.025, 0.975])
            rows.append({
                "regime": regime, "ablation": abl_name, "dataset": "CV5",
                "auc": auc_cv, "ci_lo": ci_cv[0], "ci_hi": ci_cv[1],
                "n": len(yt), "r": int(yt.sum()),
            })

            for ds_name, tdf in test_dict.items():
                Xe = tdf[feats].values.astype(float)
                ye = tdf["ruptured"].astype(int).values
                pe = clf.predict_proba(sc.transform(Xe))[:, 1]
                auc_ext = roc_auc_score(ye, pe)
                boot_ext = strat_boot_auc(ye, pe, N_BOOT, rng)
                ci_ext = np.quantile(boot_ext, [0.025, 0.975])
                rows.append({
                    "regime": regime, "ablation": abl_name, "dataset": ds_name,
                    "auc": auc_ext, "ci_lo": ci_ext[0], "ci_hi": ci_ext[1],
                    "n": len(ye), "r": int(ye.sum()),
                })

    df_out = pd.DataFrame(rows)

    print(f"{'Regime':6s} {'Ablation':10s} {'Dataset':10s} "
          f"{'AUC':>6s} {'CI_lo':>6s} {'CI_hi':>6s} {'N':>4s} {'R':>3s}")
    print("-" * 60)
    for _, r in df_out.iterrows():
        print(f"{r['regime']:6s} {r['ablation']:10s} {r['dataset']:10s} "
              f"{r['auc']:6.3f} {r['ci_lo']:6.3f} {r['ci_hi']:6.3f} "
              f"{r['n']:4.0f} {r['r']:3.0f}")

    df_out.to_csv(OUT_DIR / "M6b_bootstrap_factorial_grid.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M6b_bootstrap_factorial_grid.csv'}")


if __name__ == "__main__":
    main()
