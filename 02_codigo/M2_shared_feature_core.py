"""
M2 — Shared Feature Core harmonization
=======================================
Produz uma tabela unica com features morfologicos identicos entre AneuX e CMHA.

Fontes:
  - AneuX: 01_datasets/AneuX/data/morpho-per-cut.csv + clinical.csv
  - CMHA:  01_datasets/CMHA/statistical_results/statistical results/
           morphological_aneurysm_artery.csv + clinical_all.csv

Saida:
  - 03_analises/M2_shared_feature_core.csv

Shared core (11 features diretamente compativeis, via Raghavan 2005 / Ma 2004):
  AR, BF, CP, EI, NSI, UI, Dmax, Dn (neck width), H, S (surface), V (volume)

Harmonizacao de IDs AneuX (96.7% -> 99.7% match):
  - morpho-per-cut.csv usa sufixos de variante (_1, _2, _sib, _a, _1a, _asib)
    que clinical.csv nao usa. Resolvemos via geracao de candidatos priorizando
    exato > letra trailing > _N > sibling.
  - aneurisk usa trailing letter (C0007 clinical vs C0007a dome).
  - 2 casos genuinamente sem morfologia: SNF00000049_02 e _03.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"Y:/doutorado_noel")
ANEUX_DIR = ROOT / "01_datasets" / "AneuX" / "data"
CMHA_DIR = ROOT / "01_datasets" / "CMHA" / "statistical_results" / "statistical results"
OUT_DIR = ROOT / "03_analises"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHARED_FEATURES = ["AR", "BF", "CP", "EI", "NSI", "UI",
                   "Dmax", "Dn", "H", "S", "V"]


# -------------------------------------------------------------------
# AneuX
# -------------------------------------------------------------------
def load_aneux() -> pd.DataFrame:
    morpho = pd.read_csv(ANEUX_DIR / "morpho-per-cut.csv", header=[0, 1, 2])

    # As 3 primeiras colunas sao ('type','group','index'), ('Unnamed_1...'), ('Unnamed_2...')
    # e sua primeira LINHA contem os labels reais: 'source', 'dataset', 'cutType'.
    id_cols = morpho.columns[:3]
    id_values = morpho.iloc[0, :3].tolist()  # ['source','dataset','cutType']

    # Descarta linha-0 (cabecalho residual) e demais linhas "source"
    morpho = morpho[morpho[id_cols[0]] != "source"].copy()

    # Mapa feature -> coluna tupla
    feat_col = {
        "AR":   ("gi", "shape", "AR"),
        "BF":   ("gi", "shape", "BF"),
        "CP":   ("gi", "shape", "CP"),
        "EI":   ("gi", "shape", "EI"),
        "NSI":  ("gi", "shape", "NSI"),
        "UI":   ("gi", "shape", "UI"),
        "Dmax": ("gi", "size", "Dmax"),
        "Dn":   ("gi", "size", "Dn"),
        "H":    ("gi", "size", "H"),
        "S":    ("gi", "size", "S"),
        "V":    ("gi", "size", "V"),
    }

    core = pd.DataFrame({
        "source":  morpho[id_cols[0]].values,
        "dataset": morpho[id_cols[1]].values,
        "cutType": morpho[id_cols[2]].values,
    })
    for f, col in feat_col.items():
        core[f] = pd.to_numeric(morpho[col].values, errors="coerce")

    # As features gi.shape/gi.size so existem em 'dome' e 'ninja' (cut1/cut2
    # sao planos de visualizacao sem morfometria). Usamos 'dome' como canonico:
    # um registro por aneurisma, com as features globais integradas.
    core = core[core["cutType"] == "dome"].copy().reset_index(drop=True)

    # -----------------------------------------------------------------
    # Resolucao de nomes: clinical.vesselFileID vs morpho.dataset tem
    # variantes de sufixo. Geramos candidatos priorizando o match exato
    # e depois variantes de isolacao ou sibling. Match: 748/750 (99.7%).
    # -----------------------------------------------------------------
    clinical = pd.read_csv(ANEUX_DIR / "clinical.csv")

    dome_idx = {(r["source"], r["dataset"]): i for i, r in core.iterrows()}

    def _candidates(source: str, vid: str):
        yield vid
        if source == "aneurisk":
            yield vid + "a"
            yield vid + "b"
            return
        # hug2016 / hug2016snf / aneurist
        yield vid + "_a"
        yield vid + "_b"
        yield vid + "_sib"
        yield vid + "_asib"
        for n in range(1, 10):
            yield f"{vid}_{n}"
            yield f"{vid}_{n}a"
            yield f"{vid}_{n}b"
            yield f"{vid}_{n}sib"
            yield f"{vid}_{n}asib"
        for n in range(1, 5):
            for mn in range(1, 5):
                yield f"{vid}_{n}_{mn}"

    def _resolve(row) -> int | None:
        for cand in _candidates(row["source"], row["vesselFileID"]):
            key = (row["source"], cand)
            if key in dome_idx:
                return dome_idx[key]
        return None

    clinical["_core_idx"] = clinical.apply(_resolve, axis=1)

    n_matched = clinical["_core_idx"].notna().sum()
    print(f"  [id-resolver] AneuX clinical -> dome matched: "
          f"{n_matched}/{len(clinical)} ({100*n_matched/len(clinical):.1f}%)")

    # Attach features via indice resolvido
    feat_only = core[SHARED_FEATURES].reset_index(drop=True)
    # reindex do core pelos indices resolvidos (NaN para unmatched)
    idx = clinical["_core_idx"]
    core_subset = feat_only.reindex(idx.fillna(-1).astype(int).values)
    core_subset.index = clinical.index
    # mascarar onde nao houve match
    mask_unmatched = idx.isna()
    core_subset.loc[mask_unmatched, :] = pd.NA

    merged = pd.concat([clinical.drop(columns=["_core_idx"]), core_subset], axis=1)

    # Normalizar esquema de saida
    out = pd.DataFrame({
        "cohort":       "AneuX",
        "subcohort":    merged["source"],
        "aneurysm_id":  merged["vesselFileID"],
        "patient_id":   merged["patientID"],
        "location":     merged["location"],
        "side":         merged["side"],
        "sex":          merged["sex"],
        "age":          merged["age"],
        "ruptured":     merged["status"].map({"ruptured": 1, "unruptured": 0}),
    })
    for f in SHARED_FEATURES:
        out[f] = merged[f]
    return out


# -------------------------------------------------------------------
# CMHA
# -------------------------------------------------------------------
def load_cmha() -> pd.DataFrame:
    morpho = pd.read_csv(CMHA_DIR / "morphological_aneurysm_artery.csv")
    clinical = pd.read_csv(CMHA_DIR / "clinical_all.csv")
    # Primeira linha do clinical e cabecalho SNOMED
    clinical = clinical[clinical["number"].notna()].copy().reset_index(drop=True)

    # Renomear colunas CMHA para o esquema shared
    rename = {
        " The ratio of H to NW(AR)": "AR",
        "BottleneckFactor":           "BF",
        "ConicityParameter":          "CP",
        "EllipticityIndex":           "EI",
        "NonSphericityIndex":         "NSI",
        "UndulationIndex":            "UI",
        "D_max":                      "Dmax",
        "Neck width (NW)":            "Dn",
        "Aneurysm perpendicular height (H)": "H",
        "AneurysmSurfaceArea":        "S",
        "Aneurysm volume (V)":        "V",
    }
    morpho = morpho.rename(columns=rename)

    # Expandir 'number' do morpho: ids tipo "AHMU1218050_1" => base AHMU1218050
    # com ordinal 1. Ids sem sufixo numerico => ordinal 0 (unico).
    def _split_id(s):
        import re as _re
        s = str(s)
        m = _re.match(r"^(.+)_(\d+)$", s)
        if m:
            return m.group(1), int(m.group(2))
        return s, 0
    morpho["_base"], morpho["_ord"] = zip(*morpho["number"].map(_split_id))

    # Para clinical com linhas duplicadas (mesmo 'number'), atribuir ordinal
    # conforme ordem de aparicao (1, 2, ...). Nao-duplicadas => ordinal 0.
    clinical["_base"] = clinical["number"].astype(str)
    clinical["_occ"] = clinical.groupby("_base").cumcount() + 1  # 1,2,...
    n_per_group = clinical.groupby("_base")["_occ"].transform("max")
    clinical["_ord"] = clinical["_occ"].where(n_per_group > 1, 0)
    clinical = clinical.drop(columns=["_occ"])

    core = morpho[["_base", "_ord"] + SHARED_FEATURES].copy()

    # Normalizar clinical
    # Rupture: 1 = ruptured, 0 = unruptured
    clin = pd.DataFrame({
        "number":     clinical["number"].astype(str),
        "patient_id": clinical["number"].astype(str),
        "sex":        clinical["Gender"].map({"F": "female", "M": "male"}),
        "age":        pd.to_numeric(clinical["Age"], errors="coerce"),
        "ruptured":   pd.to_numeric(clinical["Rupture"], errors="coerce").astype("Int64"),
        "location":   clinical["location"],
        "_base":      clinical["_base"],
        "_ord":       clinical["_ord"],
    })

    merged = clin.merge(core, on=["_base", "_ord"], how="left")

    n_have = merged[SHARED_FEATURES[0]].notna().sum()
    # contar so entre aneurismas (excluir controles: ruptured NaN significa
    # Has aneurysm == 0; controles nao tem morfologia por design)
    n_aneur = merged["ruptured"].notna().sum()
    n_aneur_ok = merged[merged["ruptured"].notna()][SHARED_FEATURES[0]].notna().sum()
    print(f"  [id-resolver] CMHA clinical aneur -> morpho matched: "
          f"{n_aneur_ok}/{n_aneur} ({100*n_aneur_ok/max(n_aneur,1):.1f}%)")

    out = pd.DataFrame({
        "cohort":       "CMHA",
        "subcohort":    "CMHA",
        "aneurysm_id":  merged["number"],
        "patient_id":   merged["patient_id"],
        "location":     merged["location"],
        "side":         pd.NA,
        "sex":          merged["sex"],
        "age":          merged["age"],
        "ruptured":     merged["ruptured"],
    })
    for f in SHARED_FEATURES:
        out[f] = merged[f]
    return out


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    print("Carregando AneuX...")
    aneux = load_aneux()
    print(f"  AneuX: {len(aneux)} aneurismas")
    print(f"  rupture: {aneux['ruptured'].value_counts(dropna=False).to_dict()}")
    print(f"  subcohort: {aneux['subcohort'].value_counts().to_dict()}")
    print(f"  missing em features shared:")
    for f in SHARED_FEATURES:
        miss = aneux[f].isna().sum()
        print(f"    {f:5s}  missing={miss}")

    print("\nCarregando CMHA...")
    cmha = load_cmha()
    print(f"  CMHA: {len(cmha)} aneurismas")
    print(f"  rupture: {cmha['ruptured'].value_counts(dropna=False).to_dict()}")
    print(f"  missing em features shared:")
    for f in SHARED_FEATURES:
        miss = cmha[f].isna().sum()
        print(f"    {f:5s}  missing={miss}")

    combined = pd.concat([aneux, cmha], ignore_index=True)
    out_path = OUT_DIR / "M2_shared_feature_core.csv"
    combined.to_csv(out_path, index=False)
    print(f"\n=> {out_path}  ({len(combined)} linhas)")

    # Resumo descritivo por coorte
    print("\n--- Distribuicao das features por coorte ---")
    summary = combined.groupby("cohort")[SHARED_FEATURES].agg(["mean", "std", "count"])
    # imprimir resumido
    for f in SHARED_FEATURES:
        print(f"  {f}:")
        print(f"    AneuX: mean={summary.loc['AneuX', (f,'mean')]:.3f}  "
              f"std={summary.loc['AneuX', (f,'std')]:.3f}  "
              f"n={int(summary.loc['AneuX', (f,'count')])}")
        print(f"    CMHA:  mean={summary.loc['CMHA',  (f,'mean')]:.3f}  "
              f"std={summary.loc['CMHA',  (f,'std')]:.3f}  "
              f"n={int(summary.loc['CMHA',  (f,'count')])}")


if __name__ == "__main__":
    main()
