"""
M5 — Figuras principais do paper.

Fig 1: AUC por tercil de extremidade (HUG-trained vs pex-trained, ablacao C)
       com bootstrap CIs. Barplot side-by-side + interacao annotation.
Fig 2: ROC curves — 2 regimes x 3 datasets (6 curvas, 2 paineis).
Fig 3: Calibration plots — 2 regimes x 3 datasets (histograma de probabilidades
       + reliability curve).
Fig 4: Cohen's d heatmap — separacao ruptured/unruptured dentro de cada coorte.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"
OUT_DIR = ROOT / "03_analises" / "figures"
OUT_DIR.mkdir(exist_ok=True)

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = {"AR", "BF", "CP", "H", "Dn"}
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]

RNG = np.random.default_rng(42)
N_BOOT = 1000

COLORS = {"HUG-trained": "#D62728", "pex-trained": "#1F77B4"}
DATASET_COLORS = {
    "Internal CV5": "#2CA02C",
    "AneuX pex MCA": "#FF7F0E",
    "AneuX HUG MCA": "#FF7F0E",
    "CMHA MCA": "#9467BD",
}
DATASET_STYLES = {
    "Internal CV5": {"ls": "-", "lw": 2.2, "marker": "o", "ms": 0},
    "AneuX pex MCA": {"ls": "--", "lw": 2.0, "marker": "s", "ms": 0},
    "AneuX HUG MCA": {"ls": "--", "lw": 2.0, "marker": "s", "ms": 0},
    "CMHA MCA": {"ls": "-.", "lw": 2.0, "marker": "^", "ms": 0},
}


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


def fit_and_predict_all(train_df, hug, pex, cmha, feats, label):
    Xt, yt = train_df[feats].values.astype(float), train_df["ruptured"].astype(int).values
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced").fit(sc.transform(Xt), yt)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_cv = cross_val_predict(
        LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                           max_iter=5000, class_weight="balanced"),
        sc.transform(Xt), yt, cv=cv, method="predict_proba",
    )[:, 1]

    results = {}
    if label == "HUG-trained":
        results["Internal CV5"] = (yt, p_cv)
        results["AneuX pex MCA"] = (pex["ruptured"].astype(int).values,
                                     clf.predict_proba(sc.transform(pex[feats].values.astype(float)))[:, 1])
    else:
        results["Internal CV5"] = (yt, p_cv)
        results["AneuX pex MCA"] = (hug["ruptured"].astype(int).values,
                                     clf.predict_proba(sc.transform(hug[feats].values.astype(float)))[:, 1])
    results["CMHA MCA"] = (cmha["ruptured"].astype(int).values,
                            clf.predict_proba(sc.transform(cmha[feats].values.astype(float)))[:, 1])
    return results


def strat_boot_auc(y, p, n_boot=N_BOOT, rng=RNG):
    y = np.asarray(y); p = np.asarray(p)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    out = np.empty(n_boot)
    for i in range(n_boot):
        bp = rng.choice(pos, len(pos), replace=True)
        bn = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        out[i] = roc_auc_score(y[idx], p[idx])
    return out


# ────────── FIG 1: TERCIL EXTREMIDADE ──────────
def fig1_tercile(hug, pex, cmha):
    feats = NON_NPS
    cmha_c = cmha.copy()

    p_hug = fit_and_predict_all(hug, hug, pex, cmha, feats, "HUG-trained")["CMHA MCA"][1]
    p_pex = fit_and_predict_all(pex, hug, pex, cmha, feats, "pex-trained")["CMHA MCA"][1]
    cmha_c["p_hug"] = p_hug
    cmha_c["p_pex"] = p_pex

    unr = cmha_c[cmha_c.ruptured == 0]
    mu = unr[FEATURES_FULL].mean()
    sd = unr[FEATURES_FULL].std(ddof=1).replace(0, 1)
    z = (cmha_c[FEATURES_FULL] - mu) / sd
    cmha_c["extrem"] = z.abs().mean(axis=1)

    rom = cmha_c[cmha_c.ruptured == 1].reset_index(drop=True)
    unr_c = cmha_c[cmha_c.ruptured == 0].reset_index(drop=True)
    cut1, cut2 = float(np.quantile(rom["extrem"], 1/3)), float(np.quantile(rom["extrem"], 2/3))

    tercile_labels = ["T1\nModerate", "T2\nMedium", "T3\nExtreme"]
    tercile_cuts = [(-np.inf, cut1), (cut1, cut2), (cut2, np.inf)]

    def get_auc_boot(p_col, rom_df, unr_df):
        aucs_pt, aucs_boot = [], []
        for lo, hi in tercile_cuts:
            mask = (rom_df["extrem"] > lo) & (rom_df["extrem"] <= hi)
            sub = rom_df[mask]
            y = np.concatenate([np.ones(len(sub)), np.zeros(len(unr_df))])
            p = np.concatenate([sub[p_col].values, unr_df[p_col].values])
            aucs_pt.append(roc_auc_score(y, p))
            aucs_boot.append(strat_boot_auc(y, p))
        return aucs_pt, aucs_boot

    hug_pt, hug_b = get_auc_boot("p_hug", rom, unr_c)
    pex_pt, pex_b = get_auc_boot("p_pex", rom, unr_c)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    w = 0.35
    for i in range(3):
        ci_h = np.quantile(hug_b[i], [0.025, 0.975])
        ci_p = np.quantile(pex_b[i], [0.025, 0.975])
        ax.bar(x[i] - w/2, hug_pt[i], w, color=COLORS["HUG-trained"], alpha=0.85,
               label="HUG-trained (Abl. C)" if i == 0 else "")
        ax.errorbar(x[i] - w/2, hug_pt[i],
                    yerr=[[hug_pt[i] - ci_h[0]], [ci_h[1] - hug_pt[i]]],
                    fmt="none", color="black", capsize=4)
        ax.bar(x[i] + w/2, pex_pt[i], w, color=COLORS["pex-trained"], alpha=0.85,
               label="pex-trained (Abl. C)" if i == 0 else "")
        ax.errorbar(x[i] + w/2, pex_pt[i],
                    yerr=[[pex_pt[i] - ci_p[0]], [ci_p[1] - pex_pt[i]]],
                    fmt="none", color="black", capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tercile_labels, fontsize=11)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_title("AUC by Morphological Extremity Tercile\n(CMHA ruptured vs. unruptured, Ablation C)",
                 fontsize=12)

    gap_h = hug_pt[2] - hug_pt[0]
    gap_p = pex_pt[2] - pex_pt[0]
    ax.annotate(f"Gap T3–T1:\nHUG {gap_h:+.2f}\npex {gap_p:+.2f}\nΔ {gap_h-gap_p:+.2f} *",
                xy=(2.35, 0.72), fontsize=9, ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig1_tercile_extremity.png", dpi=300)
    fig.savefig(OUT_DIR / "Fig1_tercile_extremity.pdf")
    plt.close(fig)
    print(f"Fig1 saved: {OUT_DIR / 'Fig1_tercile_extremity.png'}")


# ────────── FIG 2: ROC CURVES ──────────
def fig2_roc(hug, pex, cmha):
    feats = NON_NPS
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_i, (train_df, train_label) in enumerate([(hug, "HUG-trained"), (pex, "pex-trained")]):
        ax = axes[ax_i]
        results = fit_and_predict_all(train_df, hug, pex, cmha, feats, train_label)

        if train_label == "pex-trained":
            rename = {"AneuX pex MCA": "AneuX HUG MCA"}
            results_renamed = {}
            for k, v in results.items():
                results_renamed[rename.get(k, k)] = v
            results = results_renamed
            ds_order = ["Internal CV5", "AneuX HUG MCA", "CMHA MCA"]
        else:
            ds_order = ["Internal CV5", "AneuX pex MCA", "CMHA MCA"]

        for ds in ds_order:
            y, p = results[ds]
            fpr, tpr, _ = roc_curve(y, p)
            auc_val = roc_auc_score(y, p)
            style = DATASET_STYLES.get(ds, {"ls": "-", "lw": 1.5})
            ds_color = DATASET_COLORS.get(ds, "grey")
            ax.plot(fpr, tpr, label=f"{ds} (AUC={auc_val:.3f})",
                    ls=style["ls"], lw=style["lw"], color=ds_color)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("1 – Specificity", fontsize=11)
        if ax_i == 0:
            ax.set_ylabel("Sensitivity", fontsize=11)
        ax.set_title(f"{train_label} — Ablation C ({len(feats)}f)", fontsize=12)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.suptitle("ROC Curves: Two Training Regimes × Three Datasets", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_ROC_two_regimes.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig2_ROC_two_regimes.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Fig2 saved: {OUT_DIR / 'Fig2_ROC_two_regimes.png'}")


# ────────── FIG 3: CALIBRATION ──────────
def fig3_calibration(hug, pex, cmha):
    feats = NON_NPS
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for row, (train_df, train_label) in enumerate([(hug, "HUG-trained"), (pex, "pex-trained")]):
        results = fit_and_predict_all(train_df, hug, pex, cmha, feats, train_label)
        if train_label == "pex-trained":
            rename = {"AneuX pex MCA": "AneuX HUG MCA"}
            results_renamed = {}
            for k, v in results.items():
                results_renamed[rename.get(k, k)] = v
            results = results_renamed
            ds_order = ["Internal CV5", "AneuX HUG MCA", "CMHA MCA"]
        else:
            ds_order = ["Internal CV5", "AneuX pex MCA", "CMHA MCA"]

        for col, ds in enumerate(ds_order):
            ax = axes[row, col]
            y, p = results[ds]
            try:
                frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
            except ValueError:
                frac_pos, mean_pred = calibration_curve(y, p, n_bins=5, strategy="uniform")

            cal_color = DATASET_COLORS.get(ds, COLORS[train_label])
            ax.plot(mean_pred, frac_pos, "o-", color=cal_color, lw=2, markersize=6)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Mean predicted probability", fontsize=10)
            if col == 0:
                ax.set_ylabel("Observed proportion", fontsize=11)
            ax.set_title(f"{train_label}\n{ds} (n={len(y)})", fontsize=11, fontweight="bold")

            ax2 = ax.twinx()
            ax2.hist(p, bins=20, range=(0, 1), alpha=0.12, color=cal_color)
            if col == 2:
                ax2.set_ylabel("Count", fontsize=9, alpha=0.6)
            else:
                ax2.set_ylabel("")
            ax2.tick_params(axis="y", labelsize=7, labelcolor="grey")

    fig.suptitle("Calibration: Two Training Regimes × Three Datasets (Ablation C)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "Fig3_calibration.png", dpi=300)
    fig.savefig(OUT_DIR / "Fig3_calibration.pdf")
    plt.close(fig)
    print(f"Fig3 saved: {OUT_DIR / 'Fig3_calibration.png'}")


# ────────── FIG 4: COHEN'S D HEATMAP ──────────
def fig4_cohens_d_heatmap(hug, pex, cmha):
    def cohens_d(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2: return np.nan
        pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
        if pooled == 0: return np.nan
        return (x.mean() - y.mean()) / pooled

    datasets = [
        ("HUG MCA\n(14R / 104U)", hug),
        ("Pseudo-ext MCA\n(30R / 40U)", pex),
        ("CMHA MCA\n(77R / 28U)", cmha),
    ]
    d_matrix = np.zeros((len(FEATURES_FULL), len(datasets)))
    for j, (label, df) in enumerate(datasets):
        r = df[df.ruptured == 1]
        u = df[df.ruptured == 0]
        for i, f in enumerate(FEATURES_FULL):
            d_matrix[i, j] = cohens_d(r[f], u[f])

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(d_matrix, cmap="RdYlBu_r", aspect="auto", vmin=-0.5, vmax=2.0)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d[0] for d in datasets], fontsize=10)
    ax.set_yticks(range(len(FEATURES_FULL)))
    ax.set_yticklabels(FEATURES_FULL, fontsize=10)
    for i in range(d_matrix.shape[0]):
        for j in range(d_matrix.shape[1]):
            v = d_matrix[i, j]
            color = "white" if abs(v) > 1.0 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9, color=color)
    plt.colorbar(im, ax=ax, label="Cohen's d (ruptured – unruptured)")
    ax.set_title("Within-Cohort Ruptured vs. Unruptured Separation\n(Cohen's d per feature)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig4_cohens_d_heatmap.png", dpi=300)
    fig.savefig(OUT_DIR / "Fig4_cohens_d_heatmap.pdf")
    plt.close(fig)
    print(f"Fig4 saved: {OUT_DIR / 'Fig4_cohens_d_heatmap.png'}")


def main():
    hug, pex, cmha = load_all()
    print(f"Loaded: hug={len(hug)} pex={len(pex)} cmha={len(cmha)}")
    print()
    fig1_tercile(hug, pex, cmha)
    fig2_roc(hug, pex, cmha)
    fig3_calibration(hug, pex, cmha)
    fig4_cohens_d_heatmap(hug, pex, cmha)
    print(f"\nAll figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
