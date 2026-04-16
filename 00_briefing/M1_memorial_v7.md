# Memorial v7 — Post-confirmatory Reframing (calibration-first)
**Data: 2026-04-16. Substitui v6.1 apos execucao M6a-M6e.**

## STATUS
v7 integra os resultados confirmatorios (M6a-M6e, commit ace1aa0 pre-execucao)
e o double-check GPT r5. Mudanca estrutural principal: **calibracao vira pilar 1,
interacao por extremidade vira pilar 2 (mecanistico, sugestivo).**

Resultados confirmatorios:
- M6a permutation interacao: **p = 0.070 (NAO significativo)**
- M6c DeLong pex > HUG em CMHA: **p = 0.191 (NAO significativo)**
- M6d Brier HUG vs pex em CMHA: **0.564 vs 0.274, CIs separados (CONFIRMADO)**
- M6e leave-5-out stability: **100% draws > 0 (CONFIRMADO)**

A versao forte da tese ("spectrum bias confirmado") esta morta. A versao publicavel
e: **evidencia convergente consistente com spectrum bias, ancorada em colapso de
calibracao confirmado e interacao estavel mas formalmente nao significativa.**

## TITULO OPERACIONAL v7
**"Catastrophic Calibration Failure Consistent with Positive-Class Spectrum Bias:
A Factorial Comparison of Training Regimes for Morphology-Based Rupture Prediction
Across Harmonized and Out-of-Domain Cohorts"**

Alternativa mais curta:
**"Internal Optimism and External Calibration Collapse in Morphology-Based Aneurysm
Rupture Models: Evidence Consistent with Positive-Class Spectrum Bias"**

(Calibracao agora aparece no titulo. "Consistent with" mantido.)

## TESE CENTRAL v7 (pos-GPT r5)
> "Training on a cohort with a small, morphologically extreme positive class inflated
> internal discrimination (AUC 0.88) and produced substantially worse external
> calibration on deployment (Brier 0.56 vs 0.27). The overall pattern was consistent
> with positive-class spectrum bias, while the pre-specified interaction by case
> extremity was positive and stable but did not meet the confirmatory permutation
> threshold (p=0.07)."

**Headline:** treino com classe positiva extrema (N=14R HUG) produz overconfidence
catastrofica no deploy externo. pex-trained (N=30R) nao e "bom" — e **menos ruim**.
A comparacao entre regimes e o produto, nao um modelo vencedor.

## DIFERENCAS vs v6.1
| Dimensao | v6.1 | v7 |
|---|---|---|
| Pilar 1 | Interacao M4i (CI exclui zero) | **Calibracao M6d** (Brier CIs separados) |
| Pilar 2 | Grid fatorial + efeitos | **Interacao** (estavel M6e 100%, mas M6a p=0.07) |
| Interacao | "formalmente robusta" | "estavel mas nao formalmente significativa" |
| pex | "mais transportavel" | "menos ruim" (NAO bom) |
| DeLong | nao testado | M6c p=0.19 — pex NAO formalmente superior em AUC |
| P(A1) | 45-55% | **35-45%** |
| Two-phase | "locked analysis plan" | **"locked follow-up analysis"** (nao pre-registration) |
| Titulo | sem calibracao | **calibracao no titulo** |

## HIERARQUIA DE EVIDENCIA v7

### Pilar 1 — Colapso de calibracao (M6d, CONFIRMADO)
**O resultado mais forte de toda a analise.**
- Brier CMHA abl-C: HUG **0.564** [0.516, 0.605] vs pex **0.274** [0.252, 0.296]
- CIs completamente separados — nenhuma sobreposicao
- Intercept CMHA: HUG 0.875 [0.754, 0.988] — overconfidence massiva
- Slope CMHA: HUG 0.047 [0.007, 0.084] vs pex 0.215 [0.032, 0.451]

HUG-trained nao so erra mais — erra com confianca alta. Isso e clinicamente
perigoso e retorica e empiricamente mais forte que qualquer diferenca de AUC.

### Pilar 2 — Gap de AUC interno (M6b, ROBUSTO)
- HUG-CV 0.882 [0.813, 0.943] vs pex-CV 0.661 [0.532, 0.791]
- CIs nao se tocam — unico efeito de AUC formalmente robusto
- Este gap e **o sintoma**: AUC interna inflada por fenotipos extremos no treino

### Pilar 3 — Interacao por extremidade (SUGESTIVO, nao confirmado)
**Exploratorio (M4i):** interacao +0.244 [+0.079, +0.403], P(diff>0)=0.998
**Confirmatorio (M6a):** permutation p=0.070 (NAO significativo)
**Estabilidade (M6e):** 100% dos 500 draws > 0, range [+0.207, +0.295]

A interacao e consistente, estavel e direcional — mas nao cruza o threshold formal.
No paper: "positive and stable but did not reach formal significance (p=0.07)".

**M4i vs M6a nao sao contraditorios:** M4i bootstrappa o gap (resampling cases),
M6a permuta labels entre modelos (mais conservador, destroi estrutura real).
No supplementary: 1 frase explicando a diferenca.

### Pilar 4 — Mecanismo extremes-overlap (M4g/M4h, DESCRITIVO)
- HUG-trained rho = +0.434 com extremidade dos rompidos CMHA (p=0.0001)
- pex-trained rho = -0.004 (p=0.97)
- HUG-trained out-of-model rho = +0.570 (Mahalanobis NPS, p<0.0001) — refuta circularidade
- AUC por tercil: HUG so funciona em extremos (T3=0.84, T1=0.61), pex uniforme

### Evidencia de suporte
- Cohen's d rompidos HUG: AR +1.33, BF +1.52, EI +1.45, NSI +1.34 (extremos)
- Site shift refutado (M4c: 0/11 features)
- HUG1-only: extremos persistem sem snf (M5e: d = +1.56, +1.74)
- Pooled piora (M5c: 0.674 < 0.697) — mais dados nao ajudam com positivos nao-representativos
- DeLong nao significativo (M6c: p=0.19) — pex nao e formalmente superior em AUC

## GRID FATORIAL 2×2 CONFIRMATORIO (M6b)

| Celula | Regime | Ablacao | CV5 [CI] | cross_ext [CI] | CMHA [CI] |
|---|---|---|---|---|---|
| HUG × full-11 | HUG (14R/118) | 11f | 0.886 [.819,.945] | 0.588 [.450,.726] | 0.640 [.513,.748] |
| HUG × sem-NPS | HUG (14R/118) | 6f | 0.882 [.813,.943] | 0.627 [.489,.755] | 0.653 [.535,.763] |
| pex × full-11 | pex (30R/70) | 11f | 0.657 [.517,.790] | 0.670 [.498,.828] | 0.592 [.448,.727] |
| pex × sem-NPS | pex (30R/70) | 6f | 0.661 [.532,.791] | 0.746 [.603,.870] | 0.697 [.584,.806] |

Efeito-regime cross: +0.101 (pex melhor). Efeito-ablacao: +0.06 (sem NPS melhor).
Interacao: pex beneficia 8× mais de remover NPS (+0.105 vs +0.013).

## CALIBRACAO COMPLETA (M6d)

| Regime | Ablacao | Dataset | Brier [CI] | Slope [CI] | Intercept [CI] |
|---|---|---|---|---|---|
| HUG | full-11 | cross_ext | 0.328 [.259,.403] | 0.022 [-.021,.068] | 0.458 [.401,.535] |
| HUG | full-11 | CMHA | 0.577 [.532,.619] | 0.064 [.024,.098] | 0.927 [.804,1.040] |
| HUG | sem-NPS | cross_ext | 0.290 [.226,.359] | 0.044 [-.004,.090] | 0.486 [.422,.561] |
| HUG | sem-NPS | CMHA | 0.564 [.516,.605] | 0.047 [.007,.084] | 0.875 [.754,.988] |
| pex | full-11 | cross_ext | 0.261 [.221,.302] | 0.075 [.015,.133] | 0.117 [.099,.134] |
| pex | full-11 | CMHA | 0.272 [.247,.300] | 0.058 [-.092,.222] | 0.740 [.726,.773] |
| pex | sem-NPS | cross_ext | 0.235 [.201,.270] | 0.108 [.041,.176] | 0.126 [.111,.149] |
| pex | sem-NPS | CMHA | 0.274 [.252,.296] | 0.215 [.032,.451] | 0.793 [.741,.871] |

## NARRATIVA DO PAPER v7

### Estrutura proposta
1. **Intro:** external validation gap, calibration neglected, spectrum bias underrecognized
2. **Methods:** two cohorts harmonized, 2×2 factorial (regime × ablation), locked follow-up
3. **Results:**
   - Liderar com gap interno (0.88 vs 0.66, CIs separados) — o sintoma
   - Grid fatorial descritivo (Tabela 2)
   - **Calibracao como resultado central** (Tabela 3 ou Figura principal)
   - Interacao por extremidade como evidencia mecanistica sugestiva
   - DeLong nao significativo (honestidade)
   - Leave-5-out como robustez
4. **Discussion:** spectrum bias como interpretacao mais consistente (nao unica),
   clinical implications of overconfident predictions, limitations (same-cohort
   confirmatory, N=14 ruptured)

### Frases-chave para o paper
- "The HUG-trained model showed not only lower external AUC but catastrophic
  calibration failure (Brier 0.56 vs 0.27, non-overlapping 95% CIs)."
- "The pre-specified interaction did not meet the confirmatory threshold
  (permutation p=0.070), although its direction was stable across all 500
  leave-5-out perturbations."
- "These findings are consistent with positive-class spectrum bias, where a
  small number of morphologically extreme ruptures in the training set inflates
  internal discrimination without producing a generalizable signal."

## VIA A2 — STATUS v7
**Prioridade mantida.** M4g/M4h sao agora pilar 3 (sugestivo), nao pilar 1.
Sem Via A2, CMHA desce de "stress cohort validada" para "stress cohort com
ressalva". Isso enfraquece a metade mecanistica mas nao mata o paper (pilar 1
calibracao sobrevive independente de CMHA admissibility).

App Via A2 pronto. Aguarda ngrok + Noel.

## KILL CRITERIOS v7
| ID | Regra | Status |
|---|---|---|
| K1 | Via A2: se CMHA rompidos pos-procedimento > 80% → caveat severo | PENDENTE Noel |
| K2 | HUG denominador | RESOLVIDO |
| K3 | Shared core <10 features | RESOLVIDO (11) |
| K4 | >5% sem patient_id | RESOLVIDO |
| K5 | PointNet++ | ADIADO |
| K6 | Ambos regimes colapsam (AUC <0.55 em ambos externos) | NAO disparado |
| K7 | <50 AneuX MCA external | OK (n=70) |
| K8 | M4i interacao CI cruza zero | NAO disparado (exploratorio OK, confirmatorio p=0.07) |

## P(A1) v7
- **Base case: 35-45%** (rebaixado de 45-55%)
- Bom caso (Via A2 completa + reframing calibracao bem comunicado): **45-50%**
- Ruim (revisor le p=0.07 como "nao ha efeito" e descarta mecanismo): **25-35%**

Razao do rebaixamento: M6a nao confirmou o teste primario. Mas M6d compensa
parcialmente — colapso de calibracao e argumento clinicamente mais forte que
interacao por tercil.

## TRANSPARENCIA / OSF v7
**"Locked follow-up analysis"** — NAO "pre-registered confirmatory study".

Descricao no paper:
> "The exploratory phase (M3–M5e) identified the spectrum bias pattern. We then
> specified five confirmatory tests with exact hypotheses and decision criteria,
> committed the analysis scripts to the repository (hash ace1aa0) before execution,
> and ran them without modification. Two of three binary-decision tests confirmed
> their hypotheses (calibration and stability), while the primary interaction
> test showed a consistent but formally non-significant effect (p=0.07)."

**NAO esconder M6a p=0.07.** Reportar no abstract ou pelo menos no results.
Honestidade aqui e diferencial competitivo — nenhum paper de aneurisma neste
campo tem locked follow-up analysis.

## CRONOGRAMA v7 (4-5 meses)
| Mes | Milestone |
|---|---|
| M1 [agora] | Memorial v7 congelado. Via A2 com Noel. Commit M6 results. |
| M2 | Draft Methods + Results (calibracao como central) |
| M3 | Draft completo + cover letter |
| M4 | Submissao EurRadiol |
| M5 | Cascade (se desk reject) — AJNR |

## DUVIDAS RESIDUAIS v7
1. Cover letter: "we found the planned cascade was non-monotonic and the
   investigation became the paper" — honesto ou suspeito? (GPT diz OK)
2. Tabela 1: incluir M4h Mahalanobis ou so mean(|z|)?
3. Figuras: Fig calibracao deve ser Figure 2 (antes da interacao)?
   Reordenar para calibration > interaction?
4. Supplementary: M4i vs M6a discrepancia — 1 frase suficiente?

## ARQUIVOS
- Memorial v6.1 (superseded): `00_briefing/M1_memorial_v6.md`
- Confirmatory plan: `00_briefing/M6_confirmatory_plan.md`
- Scripts M6a-M6e: `02_codigo/M6*.py`
- Resultados M6a-M6e: `03_analises/M6*.csv`
- Briefing GPT r5: `C:\Users\oncol\Desktop\briefing_m6_confirmatory_doublecheck.md`
- Repo: github.com/RafaelBotan/aneurysm-training-regime-comparison_phd_noel
