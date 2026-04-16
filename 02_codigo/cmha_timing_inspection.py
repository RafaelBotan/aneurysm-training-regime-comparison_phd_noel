"""
CMHA timing inspection — adjudicacao K1 gate
Via A: inspecionar DICOM + estrutura de arquivos das 77 CTAs rupturadas
procurando proxies radiologicos/metadados de timing pre vs pos-SAH.

Proxies procurados:
1. DICOM StudyDate vs AdmissionDate (se disponivel)
2. Presenca de CT simples pareado por paciente
3. SeriesDescription mencionando "SAH", "hemorrhage", "post-op"
4. Multiplos estudos por paciente (sugere seguimento)
5. Reconstrucao pos-clipagem (artefato metalico visivel em metadata talvez)

Output:
- cmha_timing_adjudication.csv: por paciente, resumo de proxies encontrados
- cmha_timing_report.md: sumario executivo com %pre / %pos / %indeterminado

REQUISITOS (instalar depois do download):
    pip install pydicom pandas
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

CMHA_ROOT = Path("Y:/doutorado_noel/01_datasets/CMHA")
PATIENTS_DIR = CMHA_ROOT / "patients_extracted"  # apos unrar
OUT_DIR = CMHA_ROOT / "timing_audit"
OUT_DIR.mkdir(exist_ok=True)

def survey_structure():
    """Primeira passada: entender estrutura antes de parsear DICOM."""
    if not PATIENTS_DIR.exists():
        print(f"[ABORT] {PATIENTS_DIR} nao existe. Rodar extracao primeiro.")
        sys.exit(1)

    patients = sorted([p for p in PATIENTS_DIR.iterdir() if p.is_dir()])
    print(f"Pacientes encontrados: {len(patients)}")

    structure = defaultdict(list)
    for p in patients[:5]:
        for item in p.rglob("*"):
            rel = item.relative_to(p)
            depth = len(rel.parts)
            if depth <= 3:
                structure[p.name].append(str(rel))

    print("\n=== Amostra de estrutura (primeiros 5 pacientes) ===")
    for pid, files in structure.items():
        print(f"\n{pid}:")
        for f in files[:20]:
            print(f"  {f}")
        if len(files) > 20:
            print(f"  ... ({len(files) - 20} mais)")

    return patients

def inspect_dicom_metadata(patients):
    """Segunda passada: extrair DICOM tags relevantes."""
    try:
        import pydicom
    except ImportError:
        print("[SKIP] pydicom nao instalado. pip install pydicom")
        return None

    import pandas as pd
    rows = []
    for p in patients:
        dcm_files = list(p.rglob("*.dcm"))[:3]
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            rows.append({
                "patient_id": p.name,
                "StudyDate": getattr(ds, "StudyDate", ""),
                "SeriesDescription": getattr(ds, "SeriesDescription", ""),
                "StudyDescription": getattr(ds, "StudyDescription", ""),
                "Modality": getattr(ds, "Modality", ""),
                "BodyPartExamined": getattr(ds, "BodyPartExamined", ""),
                "n_series_dirs": len([d for d in p.rglob("*") if d.is_dir()]),
            })
        except Exception as e:
            rows.append({"patient_id": p.name, "error": str(e)[:80]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dicom_metadata_survey.csv", index=False)
    print(f"\n[OK] Metadata salvo em {OUT_DIR / 'dicom_metadata_survey.csv'}")
    return df

def proxy_scoring(df, clinical_csv):
    """Terceira passada: scorear pre/pos por paciente ruptured."""
    import pandas as pd
    clinical = pd.read_csv(clinical_csv, encoding="utf-8-sig", skiprows=[1])
    ruptured = clinical[clinical["Rupture"] == 1]["﻿number"].tolist()
    print(f"\nRuptured candidates: {len(ruptured)}")

    report = []
    for pid in ruptured:
        row = df[df["patient_id"] == pid]
        if row.empty:
            report.append({"patient_id": pid, "verdict": "MISSING"})
            continue
        desc = " ".join([
            str(row["SeriesDescription"].iloc[0] or ""),
            str(row["StudyDescription"].iloc[0] or ""),
        ]).lower()
        verdict = "INDETERMINATE"
        evidence = []
        if any(k in desc for k in ["sah", "hemorrh", "post-op", "clip", "coil"]):
            verdict = "LIKELY_POST"
            evidence.append("study/series description")
        if "simple" in desc or "ct native" in desc or "ncct" in desc:
            evidence.append("paired non-contrast CT found")
        report.append({
            "patient_id": pid,
            "verdict": verdict,
            "evidence": "; ".join(evidence) or "none",
        })

    rpt_df = pd.DataFrame(report)
    rpt_df.to_csv(OUT_DIR / "cmha_timing_adjudication.csv", index=False)
    counts = rpt_df["verdict"].value_counts().to_dict()
    total = len(rpt_df)
    with open(OUT_DIR / "cmha_timing_report.md", "w", encoding="utf-8") as f:
        f.write(f"# CMHA Timing Adjudication — K1 Gate\n\n")
        f.write(f"Total ruptured inspected: {total}\n\n")
        for v, n in counts.items():
            f.write(f"- {v}: {n} ({100*n/total:.1f}%)\n")
        f.write(f"\n**K1 threshold:** indeterminado >20% = disparar.\n")
    print(f"\n[OK] Relatorio em {OUT_DIR / 'cmha_timing_report.md'}")
    return rpt_df

if __name__ == "__main__":
    patients = survey_structure()
    clinical_csv = CMHA_ROOT / "statistical_results" / "statistical results" / "clinical_all.csv"
    df = inspect_dicom_metadata(patients)
    if df is not None and clinical_csv.exists():
        proxy_scoring(df, clinical_csv)
