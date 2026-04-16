"""
M4i — Bootstrap gap tercil (Q2 GPT).

Bootstrap simples (1000 resamples) do gap AUC T3_extremo - T1_moderado para:
  - HUG-trained em CMHA (ablacao C)
  - pex-trained em CMHA (ablacao C)
  - Diferenca dos gaps (interacao modelo x tercil)

Modelos treinados UMA VEZ. Predicoes em CMHA fixas. Bootstrap so reamostra:
  - rompidos CMHA (n=77) com replacement
  - nao-rompidos CMHA (n=28) com replacement
Em cada bootstrap: ranquear rompidos por extremidade (ja computada uma vez),
dividir nos cortes-tercil pre-fixados, computar AUC por tercil.

Cortes-tercil: usa quantis 1/3 e 2/3 da extremidade observada nos 77 rompidos
ORIGINAIS (estavel entre bootstraps).
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

FEATURES_FULL = ["AR", "BF", "CP", "EI", "NSI", "UI", "Dmax", "Dn", "H", "S", "V"]
NPS = ["AR", "BF", "CP", "H", "Dn"]
NON_NPS = [f for f in FEATURES_FULL if f not in NPS]

RNG = np.random.default_rng(42)
N_BOOT = 1000


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
    Xe = test_df[feats].values.astype(float)
    sc = StandardScaler().fit(Xt)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=5000, class_weight="balanced")\
        .fit(sc.transform(Xt), yt)
    return clf.predict_proba(sc.transform(Xe))[:, 1]


def auc_by_tercile(idx_rom_boot, idx_unr_boot, p_pred, extrem, cut1, cut2,
                   y_rom_const=1, y_unr_const=0):
    """Computa AUC por tercil dado bootstrap idx."""
    p_rom_b = p_pred[idx_rom_boot]
    e_rom_b = extrem[idx_rom_boot]
    p_unr_b = p_pred_unr_global[idx_unr_boot]  # nao-rompidos: predicoes do mesmo modelo

    aucs = []
    for low, high in [(-np.inf, cut1), (cut1, cut2), (cut2, np.inf)]:
        mask = (e_rom_b > low) & (e_rom_b <= high)
        if mask.sum() < 3:
            aucs.append(np.nan)
            continue
        y = np.concatenate([np.ones(mask.sum()), np.zeros(len(idx_unr_boot))])
        p = np.concatenate([p_rom_b[mask], p_unr_b])
        try:
            aucs.append(roc_auc_score(y, p))
        except ValueError:
            aucs.append(np.nan)
    return aucs  # [T1, T2, T3]


def main():
    global p_pred_unr_global
    hug, pex, cmha = load_all()
    cmha = cmha.copy()

    # Predicoes ablacao C
    cmha["p_hug_C"] = fit_predict(hug, cmha, NON_NPS)
    cmha["p_pex_C"] = fit_predict(pex, cmha, NON_NPS)

    # Extremidade: mean(|z|) all-features (mesma definicao de M4g)
    unr = cmha[cmha.ruptured == 0]
    mu = unr[FEATURES_FULL].mean()
    sd = unr[FEATURES_FULL].std(ddof=1).replace(0, 1)
    z = (cmha[FEATURES_FULL] - mu) / sd
    cmha["extrem_all"] = z.abs().mean(axis=1)

    rom = cmha[cmha.ruptured == 1].reset_index(drop=True)
    unr_df = cmha[cmha.ruptured == 0].reset_index(drop=True)

    n_rom = len(rom)
    n_unr = len(unr_df)
    print(f"n_rompidos CMHA = {n_rom}, n_nao-rompidos CMHA = {n_unr}")
    print(f"N_BOOT = {N_BOOT}")
    print()

    # Cortes-tercil pre-fixados (estabilizam comparacao entre bootstraps)
    cut1 = float(np.quantile(rom["extrem_all"], 1/3))
    cut2 = float(np.quantile(rom["extrem_all"], 2/3))
    print(f"Cortes tercil extremidade: <{cut1:.3f} | {cut1:.3f}-{cut2:.3f} | >{cut2:.3f}")
    print()

    # Bootstrap
    extrem_arr = rom["extrem_all"].values
    p_hug_rom = rom["p_hug_C"].values
    p_pex_rom = rom["p_pex_C"].values
    p_pred_unr_global_local = {  # mapeamento por modelo
        "hug": unr_df["p_hug_C"].values,
        "pex": unr_df["p_pex_C"].values,
    }

    def auc_tercil(p_rom, p_unr, idx_rom, idx_unr):
        out = []
        e_b = extrem_arr[idx_rom]
        pr_b = p_rom[idx_rom]
        pu_b = p_unr[idx_unr]
        for low, high in [(-np.inf, cut1), (cut1, cut2), (cut2, np.inf)]:
            mask = (e_b > low) & (e_b <= high)
            if mask.sum() < 3:
                out.append(np.nan); continue
            y = np.concatenate([np.ones(mask.sum()), np.zeros(len(idx_unr))])
            p = np.concatenate([pr_b[mask], pu_b])
            try:
                out.append(roc_auc_score(y, p))
            except ValueError:
                out.append(np.nan)
        return out

    boot_h = np.full((N_BOOT, 3), np.nan)
    boot_p = np.full((N_BOOT, 3), np.nan)
    boot_gap_h = np.full(N_BOOT, np.nan)
    boot_gap_p = np.full(N_BOOT, np.nan)
    boot_diff_gap = np.full(N_BOOT, np.nan)

    for i in range(N_BOOT):
        idx_rom = RNG.integers(0, n_rom, n_rom)
        idx_unr = RNG.integers(0, n_unr, n_unr)
        a_h = auc_tercil(p_hug_rom, p_pred_unr_global_local["hug"], idx_rom, idx_unr)
        a_p = auc_tercil(p_pex_rom, p_pred_unr_global_local["pex"], idx_rom, idx_unr)
        boot_h[i] = a_h
        boot_p[i] = a_p
        if not np.isnan(a_h[0]) and not np.isnan(a_h[2]):
            boot_gap_h[i] = a_h[2] - a_h[0]
        if not np.isnan(a_p[0]) and not np.isnan(a_p[2]):
            boot_gap_p[i] = a_p[2] - a_p[0]
        if not np.isnan(boot_gap_h[i]) and not np.isnan(boot_gap_p[i]):
            boot_diff_gap[i] = boot_gap_h[i] - boot_gap_p[i]

    def ci(arr):
        a = arr[~np.isnan(arr)]
        return float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    print("=== Bootstrap AUC por tercil (HUG-trained, ablacao C) ===")
    for j, t in enumerate(["T1_moderado", "T2_medio", "T3_extremo"]):
        m, lo, hi = ci(boot_h[:, j])
        print(f"  {t:14s}  AUC={m:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    m, lo, hi = ci(boot_gap_h)
    excl = "SIM" if lo > 0 or hi < 0 else "NAO"
    p_gt0 = float(np.mean(boot_gap_h > 0))
    print(f"  Gap T3-T1     = {m:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  CI exclui 0? {excl}  P(gap>0)={p_gt0:.3f}")
    print()

    print("=== Bootstrap AUC por tercil (pex-trained, ablacao C) ===")
    for j, t in enumerate(["T1_moderado", "T2_medio", "T3_extremo"]):
        m, lo, hi = ci(boot_p[:, j])
        print(f"  {t:14s}  AUC={m:.3f}  CI=[{lo:.3f}, {hi:.3f}]")
    m, lo, hi = ci(boot_gap_p)
    excl = "SIM" if lo > 0 or hi < 0 else "NAO"
    p_gt0 = float(np.mean(boot_gap_p > 0))
    print(f"  Gap T3-T1     = {m:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  CI exclui 0? {excl}  P(gap>0)={p_gt0:.3f}")
    print()

    print("=== Interacao modelo x tercil (gap_HUG - gap_pex) ===")
    m, lo, hi = ci(boot_diff_gap)
    excl = "SIM" if lo > 0 or hi < 0 else "NAO"
    p_gt0 = float(np.mean(boot_diff_gap > 0))
    print(f"  Diff gap = {m:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  CI exclui 0? {excl}  P(diff>0)={p_gt0:.3f}")
    print()

    if lo > 0:
        print(">>> Diff gap > 0 com IC95% excluindo 0: HUG-trained depende de extremidade")
        print(">>> SIGNIFICATIVAMENTE mais que pex-trained. Mecanismo M4g robusto.")
    elif p_gt0 > 0.95:
        print(">>> Diff gap > 0 em >95% dos bootstraps. Forte mas IC marginalmente cruza.")
    else:
        print(">>> IC inclui 0. Mecanismo M4g como tendencia, nao significativa formalmente.")

    pd.DataFrame({
        "boot_idx": np.arange(N_BOOT),
        "auc_hug_T1": boot_h[:, 0], "auc_hug_T2": boot_h[:, 1], "auc_hug_T3": boot_h[:, 2],
        "auc_pex_T1": boot_p[:, 0], "auc_pex_T2": boot_p[:, 1], "auc_pex_T3": boot_p[:, 2],
        "gap_hug": boot_gap_h, "gap_pex": boot_gap_p, "diff_gap": boot_diff_gap,
    }).to_csv(ROOT / "03_analises" / "M4i_bootstrap_gap_tercil.csv", index=False)


if __name__ == "__main__":
    main()
