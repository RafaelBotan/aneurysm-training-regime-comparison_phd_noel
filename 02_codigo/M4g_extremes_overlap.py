"""
M4g — Mecanismo "extremos overlap": testar se HUG-trained acerta em CMHA porque
rompidos CMHA extremos se parecem com rompidos HUG extremos.

Teste 1 — Ranking por extremidade em CMHA:
  - Computar z-scores dos rompidos CMHA usando media+std dos nao-rompidos CMHA
  - Definir "extremeness" = media |z| nos 11 features (ou nos 5 NPS)
  - Dividir rompidos CMHA em tercis: T1 (extremos) / T2 / T3 (moderados)
  - Computar AUC HUG-trained e pex-trained separadamente em cada tercil
    (cada tercil vs todos os nao-rompidos CMHA)
  - Se HUG-trained AUC decresce muito de T1 para T3 e pex-trained e mais estavel,
    spectrum bias + extremes overlap confirmados.

Teste 2 — Correlacao entre predicao HUG-trained em CMHA e extremidade:
  - Spearman rho entre p_hug_trained e |z| do rompido CMHA
  - Idem para p_pex_trained
  - Se rho_hug >> rho_pex, modelo HUG depende desproporcionalmente de extremidade.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
IN_CSV = ROOT / "03_analises" / "M2_shared_feature_core.csv"

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = ["AR", "BF", "CP", "H", "Dn"]
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
    Xt, yt = train_df[feats].values.astype(float), train_df["ruptured"].astype(int).values
    Xe, ye = test_df[feats].values.astype(float), test_df["ruptured"].astype(int).values
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")\
        .fit(sc.transform(Xt), yt)
    p_test = clf.predict_proba(sc.transform(Xe))[:, 1]
    return p_test, ye


def main():
    hug, pex, cmha = load_all()

    # Extremidade: z-score dos rompidos CMHA usando media+std dos NAO-rompidos CMHA
    unr = cmha[cmha.ruptured == 0][FEATURES_FULL]
    mu = unr.mean()
    sd = unr.std(ddof=1).replace(0, 1)

    cmha = cmha.copy()
    # z por feature (signed)
    for f in FEATURES_FULL:
        cmha[f"z_{f}"] = (cmha[f] - mu[f]) / sd[f]

    z_cols_all = [f"z_{f}" for f in FEATURES_FULL]
    z_cols_nps = [f"z_{f}" for f in NPS]
    cmha["extrem_all"] = cmha[z_cols_all].abs().mean(axis=1)
    cmha["extrem_nps"] = cmha[z_cols_nps].abs().mean(axis=1)

    # Gerar predicoes em CMHA com HUG-trained e pex-trained (full-11 e ablacao C)
    configs = {
        "full_11": FEATURES_FULL,
        "sem_NPS": NON_NPS,
    }

    for abl_name, feats in configs.items():
        print(f"\n{'='*70}")
        print(f"=== Ablacao {abl_name} ({len(feats)} feats) ===")
        print(f"{'='*70}")

        p_hug, y_cmha = fit_predict(hug, cmha, feats)
        p_pex, _ = fit_predict(pex, cmha, feats)
        cmha[f"p_hug_{abl_name}"] = p_hug
        cmha[f"p_pex_{abl_name}"] = p_pex

        # Teste 2 (correlacao predicao-extremidade DENTRO dos rompidos CMHA)
        rom = cmha[cmha.ruptured == 1]
        rho_hug_all, p_hug_all = spearmanr(rom[f"p_hug_{abl_name}"], rom["extrem_all"])
        rho_pex_all, p_pex_all = spearmanr(rom[f"p_pex_{abl_name}"], rom["extrem_all"])
        rho_hug_nps, p_hug_nps = spearmanr(rom[f"p_hug_{abl_name}"], rom["extrem_nps"])
        rho_pex_nps, p_pex_nps = spearmanr(rom[f"p_pex_{abl_name}"], rom["extrem_nps"])

        print("\n-- Correlacao entre p_predicao e |z| (extremidade) dos rompidos CMHA --")
        print(f"  HUG-trained vs extrem_all (11f):  rho={rho_hug_all:+.3f} p={p_hug_all:.4f}")
        print(f"  pex-trained vs extrem_all (11f):  rho={rho_pex_all:+.3f} p={p_pex_all:.4f}")
        print(f"  HUG-trained vs extrem_nps  (5f):  rho={rho_hug_nps:+.3f} p={p_hug_nps:.4f}")
        print(f"  pex-trained vs extrem_nps  (5f):  rho={rho_pex_nps:+.3f} p={p_pex_nps:.4f}")

        # Teste 1 (AUC por tercil de extremidade)
        # Dividir rompidos CMHA em 3 tercis pela extremidade. Cada tercil vs TODOS
        # os nao-rompidos CMHA.
        terc_all = pd.qcut(rom["extrem_all"], 3, labels=["T1_moderado", "T2_medio", "T3_extremo"])
        rom = rom.assign(terc_all=terc_all)
        unr_cmha = cmha[cmha.ruptured == 0]

        print("\n-- AUC por tercil de extremidade (cada tercil + nao-rompidos) --")
        print(f"{'Modelo':14s} {'Tercil':14s} {'N_rom':>6s} {'N_unr':>6s} {'AUC':>8s}")
        for terc in ["T1_moderado", "T2_medio", "T3_extremo"]:
            sub = rom[rom.terc_all == terc]
            y = np.concatenate([np.ones(len(sub)), np.zeros(len(unr_cmha))])
            p_h = np.concatenate([sub[f"p_hug_{abl_name}"].values,
                                   unr_cmha[f"p_hug_{abl_name}"].values if f"p_hug_{abl_name}" in unr_cmha.columns
                                   else fit_predict(hug, unr_cmha, feats)[0]])
        # recriar com unruptured tb (ja tem na tabela inteira)
        # reescrevo abaixo:
        print("  (recomputado abaixo)")
        for model_name, p_col in [("HUG-trained", f"p_hug_{abl_name}"),
                                   ("pex-trained", f"p_pex_{abl_name}")]:
            for terc in ["T1_moderado", "T2_medio", "T3_extremo"]:
                sub_r = rom[rom.terc_all == terc]
                y_sub = np.concatenate([np.ones(len(sub_r)), np.zeros(len(unr_cmha))])
                p_sub = np.concatenate([sub_r[p_col].values, unr_cmha[p_col].values])
                auc = roc_auc_score(y_sub, p_sub)
                print(f"{model_name:14s} {terc:14s} {len(sub_r):>6d} {len(unr_cmha):>6d} {auc:>8.3f}")

        # Overall CMHA (referencia)
        auc_h_all = roc_auc_score(cmha.ruptured.values, cmha[f"p_hug_{abl_name}"])
        auc_p_all = roc_auc_score(cmha.ruptured.values, cmha[f"p_pex_{abl_name}"])
        print(f"\nReferencia CMHA total: HUG-trained AUC={auc_h_all:.3f}  "
              f"pex-trained AUC={auc_p_all:.3f}")

        # Deltas entre tercis
        aucs_h = []
        aucs_p = []
        for terc in ["T1_moderado", "T2_medio", "T3_extremo"]:
            sub_r = rom[rom.terc_all == terc]
            y_sub = np.concatenate([np.ones(len(sub_r)), np.zeros(len(unr_cmha))])
            aucs_h.append(roc_auc_score(y_sub, np.concatenate(
                [sub_r[f"p_hug_{abl_name}"].values, unr_cmha[f"p_hug_{abl_name}"].values])))
            aucs_p.append(roc_auc_score(y_sub, np.concatenate(
                [sub_r[f"p_pex_{abl_name}"].values, unr_cmha[f"p_pex_{abl_name}"].values])))
        gap_h = aucs_h[2] - aucs_h[0]
        gap_p = aucs_p[2] - aucs_p[0]
        print(f"\nGap AUC (T3_extremo - T1_moderado):")
        print(f"  HUG-trained: {gap_h:+.3f}  (se grande + POSITIVO: depende de extremidade)")
        print(f"  pex-trained: {gap_p:+.3f}")
        print(f"  Diferenca gap: {gap_h - gap_p:+.3f}  (positivo = HUG mais dependente de extremos)")


if __name__ == "__main__":
    main()
