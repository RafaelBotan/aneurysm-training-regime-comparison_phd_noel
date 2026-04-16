"""
Figure S2 — Leave-5-out interaction distribution histogram.
Reads M6e raw results and plots distribution of 500 interaction values.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises" / "figures"

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
    clf = LogisticRegression(C=1.0, solver="lbfgs",
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

    p_hug = fit_predict_cmha(hug, cmha, feats)
    p_pex = fit_predict_cmha(pex, cmha, feats)
    y = cmha["ruptured"].astype(int).values

    unr = cmha[cmha.ruptured == 0]
    mu = unr[FEATURES_FULL].mean()
    sd = unr[FEATURES_FULL].std(ddof=1).replace(0, 1)
    extrem = ((cmha[FEATURES_FULL] - mu) / sd).abs().mean(axis=1).values

    interaction_full = compute_interaction(p_hug, p_pex, y, extrem)

    rom_idx = np.where(y == 1)[0]
    interactions = np.empty(N_DRAWS)
    for i in range(N_DRAWS):
        exclude = rng.choice(rom_idx, K_OUT, replace=False)
        keep = np.ones(len(cmha), dtype=bool)
        keep[exclude] = False
        interactions[i] = compute_interaction(
            p_hug[keep], p_pex[keep], y[keep], extrem[keep])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(interactions, bins=30, color="#5B9BD5", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", ls="--", lw=1.5, label="Zero (no interaction)")
    ax.axvline(interaction_full, color="black", ls="-", lw=2,
               label=f"Full-sample ({interaction_full:+.3f})")

    frac = np.mean(interactions > 0)
    ax.set_xlabel("Interaction (gap_HUG − gap_pex)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Leave-5-out Stability: {frac*100:.0f}% of {N_DRAWS} draws > 0",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"FigS2_leave5out_histogram.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"=> {OUT_DIR / 'FigS2_leave5out_histogram.png'}")
    plt.close()


if __name__ == "__main__":
    main()
