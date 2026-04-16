"""
M6d — CONFIRMATORY (pre-registered). DO NOT RUN until plan is committed.

Formal calibration metrics with bootstrap CI per cell of the 2x2 grid
on external datasets (cross_ext + CMHA).
Metrics: Brier score, calibration slope, calibration intercept.
1000 resamples. Descriptive + decision on Brier (HUG worse if CI above pex CI).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from scipy.special import logit

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]
ABLATIONS = {"full_11": FEATURES_FULL, "sem_NPS": NON_NPS}

N_BOOT = 1000
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


def cal_slope_intercept(y, p):
    """Calibration slope and intercept via logistic calibration."""
    eps = 1e-8
    p_clip = np.clip(p, eps, 1 - eps)
    lp = logit(p_clip)
    lr = LinearRegression().fit(lp.reshape(-1, 1), y)
    return float(lr.coef_[0]), float(lr.intercept_)


def compute_metrics(y, p):
    brier = brier_score_loss(y, p)
    slope, intercept = cal_slope_intercept(y, p)
    return {"brier": brier, "cal_slope": slope, "cal_intercept": intercept}


def boot_metrics(y, p, n_boot, rng):
    y = np.asarray(y); p = np.asarray(p)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    results = {"brier": [], "cal_slope": [], "cal_intercept": []}
    for _ in range(n_boot):
        bp = rng.choice(pos, len(pos), replace=True)
        bn = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        m = compute_metrics(y[idx], p[idx])
        for k in results:
            results[k].append(m[k])
    return {k: np.array(v) for k, v in results.items()}


def main():
    rng = np.random.default_rng(SEED)
    hug, pex, cmha = load_all()

    regimes = {
        "HUG": (hug, {"cross_ext": pex, "CMHA": cmha}),
        "pex": (pex, {"cross_ext": hug, "CMHA": cmha}),
    }

    print("=" * 80)
    print("M6d — Formal calibration metrics with bootstrap CI (1000 resamples)")
    print("=" * 80)
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

            for ds_name, tdf in test_dict.items():
                Xe = tdf[feats].values.astype(float)
                ye = tdf["ruptured"].astype(int).values
                pe = clf.predict_proba(sc.transform(Xe))[:, 1]

                pt = compute_metrics(ye, pe)
                bt = boot_metrics(ye, pe, N_BOOT, rng)

                row = {
                    "regime": regime, "ablation": abl_name, "dataset": ds_name,
                    "n": len(ye), "r": int(ye.sum()),
                }
                for metric in ["brier", "cal_slope", "cal_intercept"]:
                    ci = np.quantile(bt[metric], [0.025, 0.975])
                    row[metric] = pt[metric]
                    row[f"{metric}_ci_lo"] = ci[0]
                    row[f"{metric}_ci_hi"] = ci[1]
                rows.append(row)

    df_out = pd.DataFrame(rows)

    for metric in ["brier", "cal_slope", "cal_intercept"]:
        print(f"\n--- {metric.upper()} ---")
        print(f"{'Regime':6s} {'Ablation':10s} {'Dataset':10s} "
              f"{'Value':>7s} {'CI_lo':>7s} {'CI_hi':>7s}")
        for _, r in df_out.iterrows():
            print(f"{r['regime']:6s} {r['ablation']:10s} {r['dataset']:10s} "
                  f"{r[metric]:7.3f} {r[f'{metric}_ci_lo']:7.3f} "
                  f"{r[f'{metric}_ci_hi']:7.3f}")

    # Decision: compare Brier on CMHA, ablation C
    hug_brier = df_out[(df_out.regime == "HUG") & (df_out.ablation == "sem_NPS")
                        & (df_out.dataset == "CMHA")].iloc[0]
    pex_brier = df_out[(df_out.regime == "pex") & (df_out.ablation == "sem_NPS")
                        & (df_out.dataset == "CMHA")].iloc[0]
    print(f"\n=== DECISION (Brier on CMHA, Abl. C) ===")
    print(f"HUG: {hug_brier['brier']:.3f} [{hug_brier['brier_ci_lo']:.3f}, "
          f"{hug_brier['brier_ci_hi']:.3f}]")
    print(f"pex: {pex_brier['brier']:.3f} [{pex_brier['brier_ci_lo']:.3f}, "
          f"{pex_brier['brier_ci_hi']:.3f}]")
    if hug_brier["brier_ci_lo"] > pex_brier["brier_ci_hi"]:
        print(">>> HUG Brier CI entirely above pex CI: HUG calibration WORSE.")
    elif pex_brier["brier_ci_lo"] > hug_brier["brier_ci_hi"]:
        print(">>> pex Brier CI entirely above HUG CI: pex calibration WORSE.")
    else:
        print(">>> Brier CIs overlap: no clear winner on calibration.")

    df_out.to_csv(OUT_DIR / "M6d_calibration_formal.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M6d_calibration_formal.csv'}")


if __name__ == "__main__":
    main()
