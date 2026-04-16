"""
M5d — Fatorial 2x2: regime (HUG vs pex) x ablacao (full-11 vs sem-NPS).

Correcao GPT #1: comparacao primaria nao pode confundir regime com feature set.
Grid de 4 celulas, cada uma avaliada nos 3 datasets.

Tambem reporta "efeito regime" (media das 2 ablacoes) e "efeito ablacao"
(media dos 2 regimes) para separar contribuicoes.
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

ABLATIONS = {
    "full_11": FEATURES_FULL,
    "sem_NPS": NON_NPS,
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


def eval_cell(train_df, test_dict, feats):
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
    res = {"auc_cv5": roc_auc_score(yt, p_cv)}
    for name, tdf in test_dict.items():
        Xe = tdf[feats].values.astype(float)
        ye = tdf["ruptured"].astype(int).values
        pe = clf.predict_proba(sc.transform(Xe))[:, 1]
        res[f"auc_{name}"] = roc_auc_score(ye, pe)
    return res


def main():
    hug, pex, cmha = load_all()

    print("=" * 76)
    print("M5d — FATORIAL 2×2: regime (HUG vs pex) × ablacao (full-11 vs sem-NPS)")
    print("=" * 76)
    print()

    regimes = {
        "HUG": (hug, {"cross_ext": pex, "CMHA": cmha}),
        "pex": (pex, {"cross_ext": hug, "CMHA": cmha}),
    }

    rows = []
    grid = {}
    for regime, (train_df, test_dict) in regimes.items():
        for abl, feats in ABLATIONS.items():
            cell = f"{regime}_{abl}"
            r = eval_cell(train_df, test_dict, feats)
            r["regime"] = regime
            r["ablation"] = abl
            r["n_feats"] = len(feats)
            r["n_train"] = len(train_df)
            r["r_train"] = int(train_df.ruptured.sum())
            rows.append(r)
            grid[cell] = r

    df_out = pd.DataFrame(rows)

    # Print grid
    print(f"{'Cell':20s}  {'CV5':>6s}  {'cross_ext':>10s}  {'CMHA':>6s}  {'nf':>3s}")
    print("-" * 55)
    for _, r in df_out.iterrows():
        cell = f"{r['regime']}_{r['ablation']}"
        print(f"{cell:20s}  {r['auc_cv5']:6.3f}  "
              f"{r['auc_cross_ext']:10.3f}  {r['auc_CMHA']:6.3f}  {r['n_feats']:3.0f}")
    print()

    # Efeito regime (media das ablacoes)
    print("=== Efeito REGIME (media das 2 ablacoes) ===")
    for metric in ["auc_cv5", "auc_cross_ext", "auc_CMHA"]:
        hug_mean = np.mean([grid["HUG_full_11"][metric], grid["HUG_sem_NPS"][metric]])
        pex_mean = np.mean([grid["pex_full_11"][metric], grid["pex_sem_NPS"][metric]])
        delta = pex_mean - hug_mean
        print(f"  {metric:18s}  HUG={hug_mean:.3f}  pex={pex_mean:.3f}  Δ(pex-HUG)={delta:+.3f}")
    print()

    # Efeito ablacao (media dos regimes)
    print("=== Efeito ABLACAO (media dos 2 regimes) ===")
    for metric in ["auc_cv5", "auc_cross_ext", "auc_CMHA"]:
        f11_mean = np.mean([grid["HUG_full_11"][metric], grid["pex_full_11"][metric]])
        nps_mean = np.mean([grid["HUG_sem_NPS"][metric], grid["pex_sem_NPS"][metric]])
        delta = nps_mean - f11_mean
        print(f"  {metric:18s}  full_11={f11_mean:.3f}  sem_NPS={nps_mean:.3f}  Δ(semNPS-full)={delta:+.3f}")
    print()

    # Interacao: pex ganha MAIS com sem_NPS que HUG?
    print("=== Interacao regime × ablacao (delta CMHA) ===")
    hug_delta = grid["HUG_sem_NPS"]["auc_CMHA"] - grid["HUG_full_11"]["auc_CMHA"]
    pex_delta = grid["pex_sem_NPS"]["auc_CMHA"] - grid["pex_full_11"]["auc_CMHA"]
    interaction = pex_delta - hug_delta
    print(f"  HUG: full->semNPS delta CMHA = {hug_delta:+.3f}")
    print(f"  pex: full->semNPS delta CMHA = {pex_delta:+.3f}")
    print(f"  Interacao = {interaction:+.3f}")
    if abs(interaction) > 0.05:
        print(f"  >>> Interacao nao-trivial: pex beneficia {'mais' if interaction > 0 else 'menos'} de remover NPS.")
    else:
        print(f"  >>> Interacao pequena: efeito de ablacao similar nos dois regimes.")

    df_out.to_csv(OUT_DIR / "M5d_factorial_2x2.csv", index=False)
    print(f"\n=> {OUT_DIR / 'M5d_factorial_2x2.csv'}")


if __name__ == "__main__":
    main()
