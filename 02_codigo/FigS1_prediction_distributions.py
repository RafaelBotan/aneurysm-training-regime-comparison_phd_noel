"""
Figure S1 — Prediction distributions by regime on CMHA.
Histograms of predicted probabilities split by true label.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises" / "figures"

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


def fit_predict(train_df, test_df, feats):
    Xt = train_df[feats].values.astype(float)
    yt = train_df["ruptured"].astype(int).values
    Xe = test_df[feats].values.astype(float)
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")
    clf.fit(sc.transform(Xt), yt)
    return clf.predict_proba(sc.transform(Xe))[:, 1]


def main():
    hug, pex, cmha = load_all()
    feats = NON_NPS
    y_cmha = cmha["ruptured"].astype(int).values

    p_hug = fit_predict(hug, cmha, feats)
    p_pex = fit_predict(pex, cmha, feats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    bins = np.linspace(0, 1, 21)

    for ax, preds, title in zip(axes, [p_hug, p_pex],
                                 ["HUG-trained → CMHA", "pex-trained → CMHA"]):
        ax.hist(preds[y_cmha == 0], bins=bins, alpha=0.6, color="#4472C4",
                label=f"Unruptured (n={int((y_cmha==0).sum())})", edgecolor="white")
        ax.hist(preds[y_cmha == 1], bins=bins, alpha=0.6, color="#C44E52",
                label=f"Ruptured (n={int((y_cmha==1).sum())})", edgecolor="white")
        ax.set_xlabel("Predicted probability of rupture", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.axvline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)

    axes[0].set_ylabel("Count", fontsize=12)
    fig.suptitle("Prediction Distributions on CMHA (Ablation C, 6 features)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"FigS1_prediction_distributions.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"=> {OUT_DIR / 'FigS1_prediction_distributions.png'}")
    plt.close()


if __name__ == "__main__":
    main()
