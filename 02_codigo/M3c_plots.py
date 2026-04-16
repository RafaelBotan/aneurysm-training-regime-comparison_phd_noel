"""
M3c — Figuras para paper: ROC + calibration (reliability) por ablacao.

Usa M3_predictions.csv e recomputa os preditores internos (CV5 stratified)
para gerar figuras interno vs externo.

Saida:
  03_analises/M3_fig_roc.png
  03_analises/M3_fig_calibration.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
PRED_CSV = ROOT / "03_analises" / "M3_predictions.csv"
OUT_DIR = ROOT / "03_analises"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
ABLATIONS = {
    "A_full_11":              FEATURES_FULL,
    "B_sem_CP":               [f for f in FEATURES_FULL if f != "CP"],
    "C_sem_neck_plane_sens":  [f for f in FEATURES_FULL if f not in NPS],
}


def load_splits():
    df = pd.read_csv(IN_CSV)
    tr = df[(df.cohort == "AneuX")
            & df.subcohort.isin(["hug2016", "hug2016snf"])
            & df.location.fillna("").str.contains("MCA")
            & df.ruptured.notna()].dropna(subset=FEATURES_FULL).reset_index(drop=True)
    te = df[(df.cohort == "CMHA") & df.ruptured.notna()]\
             .dropna(subset=FEATURES_FULL).reset_index(drop=True)
    return tr, te


def internal_cv_probs(train, feats):
    X = train[feats].values.astype(float)
    y = train["ruptured"].astype(int).values
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
    return y, p


# -------------------------------------------------------------------
# Figura 1: ROC internal vs external por ablacao
# -------------------------------------------------------------------
def fig_roc(train, preds_df):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (abl, feats) in zip(axs, ABLATIONS.items()):
        # Internal
        y_in, p_in = internal_cv_probs(train, feats)
        auc_in = roc_auc_score(y_in, p_in)
        fpr_in, tpr_in, _ = roc_curve(y_in, p_in)
        ax.plot(fpr_in, tpr_in, lw=2.2, label=f"AneuX HUG MCA CV5  AUC={auc_in:.3f}",
                color="#1f77b4")
        # External (train_scaled mode)
        sub = preds_df[(preds_df.ablation == abl) & (preds_df["mode"] == "train_scaled")]
        auc_ex = roc_auc_score(sub.y_test, sub.p_test)
        fpr_ex, tpr_ex, _ = roc_curve(sub.y_test, sub.p_test)
        ax.plot(fpr_ex, tpr_ex, lw=2.2, label=f"CMHA MCA ext      AUC={auc_ex:.3f}",
                color="#d62728")
        # Chance
        ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6)
        ax.set_xlabel("1 - Specificity (FPR)")
        ax.set_title({"A_full_11":              "A) full 11 features",
                      "B_sem_CP":               "B) sem CP",
                      "C_sem_neck_plane_sens":  "C) sem AR,BF,CP,H,Dn"}[abl])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Sensitivity (TPR)")
    fig.suptitle("ROC: internal AneuX HUG MCA (CV5) vs external CMHA MCA",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "M3_fig_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------
# Figura 2: Calibration / reliability diagram (CMHA external)
# -------------------------------------------------------------------
def fig_calibration(preds_df):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (abl, _) in zip(axs, ABLATIONS.items()):
        sub = preds_df[(preds_df.ablation == abl) & (preds_df["mode"] == "train_scaled")]
        y = sub.y_test.values
        p = sub.p_test.values
        # 10 bins equal-width
        bins = np.linspace(0, 1, 11)
        inds = np.digitize(p, bins) - 1
        inds = np.clip(inds, 0, 9)
        bin_mid, bin_obs, bin_pred, bin_n = [], [], [], []
        for b in range(10):
            mask = inds == b
            if mask.sum() == 0: continue
            bin_mid.append((bins[b] + bins[b + 1]) / 2)
            bin_obs.append(y[mask].mean())
            bin_pred.append(p[mask].mean())
            bin_n.append(mask.sum())
        sizes = np.array(bin_n) * 20 + 30
        ax.scatter(bin_pred, bin_obs, s=sizes, alpha=0.7, color="#d62728",
                   edgecolor="black", linewidth=0.8, zorder=3)
        ax.plot([0, 1], [0, 1], ls="--", color="grey", alpha=0.6, label="Perfect")
        ax.axhline(y.mean(), ls=":", color="#d62728", alpha=0.5,
                   label=f"CMHA prev={y.mean():.2f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_title({"A_full_11":              "A) full 11 features",
                      "B_sem_CP":               "B) sem CP",
                      "C_sem_neck_plane_sens":  "C) sem AR,BF,CP,H,Dn"}[abl])
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Observed rupture fraction")
    fig.suptitle("Calibration on CMHA MCA (train_scaled) — size ~ n in bin",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "M3_fig_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train, _ = load_splits()
    preds_df = pd.read_csv(PRED_CSV)
    fig_roc(train, preds_df)
    print(f"=> {OUT_DIR / 'M3_fig_roc.png'}")
    fig_calibration(preds_df)
    print(f"=> {OUT_DIR / 'M3_fig_calibration.png'}")


if __name__ == "__main__":
    main()
