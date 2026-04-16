# M1 Data Freeze Memo — Rupture-gen AneuX+CMHA
**Data abertura:** 2026-04-15
**Status:** EM CURSO
**Owner:** Claude Code / Sergio Arruda / Noel

Este memo congela as ambiguidades de coorte e feature dictionary antes de qualquer modelagem. Sem assinatura dos 3 gates abaixo, zero treino.

---

## Gate 1 — HUG denominator freeze

**Fato validado:** AneuX contem HUG1 = 350 casos (Geneva coorte 1) + HUG2 = 120 casos (Geneva coorte 2) = 470 casos Geneva totais.

**Ambiguidade:** memorial previo usou "HUG 350" como conjunto de desenvolvimento. Isso implica exclusao de HUG2=120 sem justificativa explicita.

**Decisao necessaria:**
- [ ] Usar HUG1+HUG2 = 470 casos (apos exclusao cavernous ICA)?
- [ ] Ou usar so HUG1 = 350? Se sim, justificativa formal documentada (diferenca de protocolo? data cut? inclusion criteria?).

**Acao:**
1. Abrir metadata AneuX content-description-v1.0.pdf (Zenodo)
2. Ler Juchler 2022 Frontiers Neurology paper (secao Methods/Cohort)
3. Verificar splits originais usados por Cao 2024 (Frontiers Physiology) — o que ele considerou HUG?
4. Se ambiguo: default = HUG1+HUG2 completo, reportar subgroup analysis HUG1-only como robustness check

**Deadline:** fim M1
**Status:** PENDENTE

---

## Gate 2 — Feature dictionary freeze

**Fato validado:** paper Juchler 2022 diz 150 indices morfometricos; Zenodo metadata diz 170.

**Ambiguidade:** divergencia real. Precisa abrir CSV para contagem definitiva.

**Decisao necessaria:**
- [ ] N real de indices disponiveis no CSV
- [ ] Subset uniformemente disponivel AneuX + CMHA (interseccao computavel)
- [ ] Lista congelada com definicao operacional de cada feature

**Acao:**
1. Baixar AneuX (6.3 GB, trivial com 1 Gbps)
2. Abrir CSV de indices, contar colunas
3. Mapear quais indices tambem estao em CMHA CFD CSV (ou computaveis a partir do Parasolid)
4. Congelar lista: nome + formula + unidade + arquivo fonte

**Kill gate K3:**
- Se interseccao <100 features geometricas utilizaveis → reduzir escopo
- Se >10% das centrais com missingness/incompatibilidade → reduzir ou abortar braco tabular

**Deadline:** fim M1
**Status:** PENDENTE (aguardando download AneuX)

---

## Gate 3 — CMHA admissibility (timing pre/pos-ruptura)

**Fato validado:** CMHA tem 77 aneurismas ruptured + 28 unruptured. Nao esta claro se as CTAs dos rupturados foram adquiridas PRE ou POS-SAH.

**Ambiguidade critica:** se pos-SAH, morfologia ja alterada pelo evento = confound "post-rupture morphology". Modelo aprende geometria deformada, nao preditor de ruptura futura.

**Decisao necessaria:**
- Determinacao direta: pre / pos / misto / indeterminado
- Se misto ou indeterminado: percentual ou range

**Acao:**
1. Agent dispatch 2026-04-15 noite: deep read Nature Sci Data s41597-024-04056-8 + suplemento + figshare metadata
2. Busca Google Scholar por papers que citam CMHA e discutem timing
3. Se necessario: email ao corresponding author (Song et al., Anhui Medical)

**Kill gate K1:**
- Indeterminado >20% dos 77 → CMHA rebaixado a stress test secundario com "apparent transportability" permanente
- Claro pos-ruptura >50% → abortar flagship ou reformular headline
- Claro pre-ruptura >80% → gate OK, remover "Apparent" do titulo

**Deadline:** fim M1
**Status:** K1 ACIONADO textualmente (2026-04-15 noite tarde). Paper Song 2024 nao menciona timing. Presuncao clinica ~100% pos-SAH mas sem evidencia textual.

**Decisao operacional 2026-04-15:** "Apparent" permanente no titulo ate adjudicacao definitiva. Duas vias paralelas para tentar remover:

**Achado 2026-04-15 (statistical_results.rar baixado + extraido):**
- clinical_all.csv (151 linhas: 77 R + 28 UR + 44 controles + 2 header) tem 19 colunas total.
- **NENHUMA coluna de timing CTA vs SAH.** Unica coluna relacionada e "Ealier SAH from another aneurysm" = historia previa de SAH de outro aneurisma, NAO timing do CTA atual.
- Interpretacao: autores nao expuseram timing no dado processado. Ou nao rastrearam, ou omitiram intencionalmente. **Evidencia negativa forte para adjudicacao textual.**
- Adjudicacao definitiva via (a) depende de DICOM StudyDate + SAH onset em prontuario (nao disponivel) OU radiologia proxy (SAH/hematoma/clip em CT simples pareado) nas 10 GB de patients.rar.
- morphological_aneurysm_artery.csv: 28 colunas (N real CMHA morphological index set — NAO confundir com divergencia 150/170 AneuX).

- **Via A metadata (2026-04-15 tarde): NEGATIVA.**
  - CMHA nao tem DICOM — tem NIfTI (`.nii.gz`) + STL + Fluent CAS + sidecar JSON BIDS (dcm2niix).
  - 99 JSONs inspecionados: todos uniformes ("HEAD CTA", "3.6 Head CTA(DSA)", "S to I"). Zero StudyDate/AcquisitionDate (de-identificacao agressiva). Zero mencao a SAH/clip/coil/hemorrh/post-op em Series/Protocol/Procedure descriptions.
  - Unica metadata temporal: AcquisitionTime (HH:MM:SS sem data) — inutil para timing pre/pos-SAH.
  - **Conclusao: adjudicacao textual + metadata esgotada. So resta radiologia visual (Via A2) ou resposta do autor (Via B).**
- **Via A2 radiologia visual:** Noel (ou detector automatizado Aidoc/Qure) inspeciona NIfTI das 77 CTAs procurando SAH, hidrocefalia, clips, coils, EVD. 1-2 dias de Noel. Conclusivo se >=50% tiver proxy claro pos-SAH.
- **Via B (email Xijun Gong):** Second Affiliated Hospital Anhui Medical. Draft em Desktop. Resposta 1-4 sem ou silencio.
- **Via C (fallback):** /double-check GPT sobre interpretacao da regra K1 se A2 e B falharem.

Se A2 e B negativos: "Apparent" fica permanente. K6 (collapse) continua valendo como kill de flagship.

---

## Entregaveis M1 (todos antes de M2)

1. [ ] Este memo com 3 gates assinados
2. [ ] data_dictionary_frozen.csv (feature names, formulas, sources, dtypes)
3. [ ] cohort_map.json (patient_id → aneurysm_id → source_dataset → split role)
4. [ ] OSF pre-registration com commit hash AneuX + access date
5. [ ] Baseline reproduzido do Cao 2024 em AneuX harmonizado (opcional M1, obrigatorio M2)

## Sign-off
- [ ] Gate 1 HUG denominator: ASSINADO / PENDENTE
- [ ] Gate 2 feature dictionary: ASSINADO / PENDENTE
- [ ] Gate 3 CMHA timing: ASSINADO / PENDENTE

**Sem os 3 assinados, zero treino.**
