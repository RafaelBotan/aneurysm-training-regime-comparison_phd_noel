"""
M6c — CONFIRMATORY (pre-registered). DO NOT RUN until plan is committed.

DeLong test: pex-trained AUC > HUG-trained AUC on CMHA (ablation C).
One-sided p < 0.05.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
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


def delong_variance(y_true, p1, p2):
    """Compute DeLong test statistic for two correlated AUCs on same sample."""
    y = np.asarray(y_true, int)
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)

    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    m = len(pos)
    n = len(neg)

    def placement_values(pred):
        V10 = np.empty(m)
        V01 = np.empty(n)
        for i, pi in enumerate(pos):
            V10[i] = np.mean(pred[pi] > pred[neg]) + 0.5 * np.mean(pred[pi] == pred[neg])
        for j, nj in enumerate(neg):
            V01[j] = np.mean(pred[pos] > pred[nj]) + 0.5 * np.mean(pred[pos] == pred[nj])
        return V10, V01

    V10_1, V01_1 = placement_values(p1)
    V10_2, V01_2 = placement_values(p2)

    S10 = np.cov(V10_1, V10_2, ddof=1)
    S01 = np.cov(V01_1, V01_2, ddof=1)

    S = S10 / m + S01 / n

    auc1 = roc_auc_score(y, p1)
    auc2 = roc_auc_score(y, p2)
    diff = auc2 - auc1

    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        var_diff = 1e-10

    z = diff / np.sqrt(var_diff)
    p_one_sided = 1.0 - norm.cdf(z)

    return auc1, auc2, diff, z, p_one_sided


def main():
    hug, pex, cmha = load_all()
    feats = NON_NPS

    Xh = hug[feats].values.astype(float)
    yh = hug["ruptured"].astype(int).values
    sc_h = StandardScaler().fit(Xh)
    clf_h = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                max_iter=5000, class_weight="balanced")
    clf_h.fit(sc_h.transform(Xh), yh)

    Xp = pex[feats].values.astype(float)
    yp = pex["ruptured"].astype(int).values
    sc_p = StandardScaler().fit(Xp)
    clf_p = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                max_iter=5000, class_weight="balanced")
    clf_p.fit(sc_p.transform(Xp), yp)

    Xc = cmha[feats].values.astype(float)
    yc = cmha["ruptured"].astype(int).values
    p_hug = clf_h.predict_proba(sc_h.transform(Xc))[:, 1]
    p_pex = clf_p.predict_proba(sc_p.transform(Xc))[:, 1]

    auc_h, auc_p, diff, z, p_val = delong_variance(yc, p_hug, p_pex)

    print("=" * 60)
    print("M6c — DeLong test: pex-trained vs HUG-trained on CMHA (Abl. C)")
    print("=" * 60)
    print(f"HUG-trained AUC on CMHA: {auc_h:.3f}")
    print(f"pex-trained AUC on CMHA: {auc_p:.3f}")
    print(f"Difference (pex - HUG):  {diff:+.3f}")
    print(f"DeLong z-statistic:      {z:+.3f}")
    print(f"One-sided p-value:       {p_val:.4f}")
    print()

    if p_val < 0.05:
        print(">>> SIGNIFICANT (p < 0.05): pex-trained AUC > HUG-trained on CMHA.")
    else:
        print(">>> NOT significant (p >= 0.05): no confirmed superiority.")

    results = pd.DataFrame([{
        "test": "delong_cmha_ablC",
        "auc_hug": auc_h,
        "auc_pex": auc_p,
        "diff_pex_minus_hug": diff,
        "z_stat": z,
        "p_one_sided": p_val,
        "significant_005": p_val < 0.05,
    }])
    results.to_csv(OUT_DIR / "M6c_delong_test.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M6c_delong_test.csv'}")


if __name__ == "__main__":
    main()
