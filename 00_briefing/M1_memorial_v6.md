# Memorial v6.1 — Spectrum-bias Reframing (com correcoes GPT r4)
**Data: 2026-04-15 fim de noite. Substitui v5 (FROZEN) e v6.0 apos auditorias M3-M5e.**

## STATUS
v6.1 integra 3 correcoes estruturais GPT r4:
1. **Comparacao primaria fatorial 2×2** (regime × ablacao), nao HUG/full vs pex/semNPS confundidos
2. **Sensitivity HUG1-only** incluida — spectrum bias NAO vem de snf (M5e confirmou)
3. **Via A2 Rota A** — M4g/M4h/M4i dependem de CMHA, entao Via A2 sobe para alta prioridade
   pre-submissao (nao gate absoluto, mas nao desce a opcional)

M4b-M4i identificaram **positive-class spectrum bias no treino HUG** como mecanismo
dominante. Bootstrap formal (M4i) mostra interacao modelo × tercil de extremidade
significativa (CI 95% exclui zero). M4h blinda contra circularidade (out-of-model).
M5d fatorial separa cleanly efeito-regime de efeito-ablacao.
M5e confirma que extremos HUG persistem sem snf.

## TITULO OPERACIONAL v6.1
**"Internal Discrimination Consistent with Positive-Class Spectrum Bias: A Factorial
Comparison of Training Regimes and Feature Sets for Morphology-Only Rupture Models
Across Harmonized and Out-of-Domain Cohorts"**

(Titulo usa "Consistent with" em vez de "Inflated by" — alinhado com evidencia mecanistica.)

## TESE CENTRAL (v6, GPT-revisada)
> "A larger overall training cohort with a small, morphologically extreme positive
> class showed higher internal discrimination but less stable external transport
> than training on a smaller yet more representative rupture phenotype.
> The high internal AUC (0.88) is consistent with positive-class spectrum bias
> rather than a generalizable rupture signal."

**Headline:** comparacao de **dois regimes de treino** — HUG-extreme (n=14R) vs
pex-representative (n=30R) — em duas coortes externas (HUG/pex-MCA e CMHA).
NAO promocao de um vencedor.

## DIFERENCAS vs v5
| Dimensao | v5 | v6 |
|---|---|---|
| Tese | "compound domain shift" | "positive-class spectrum bias inflates internal AUC" |
| Cascade | monotonica prevista | non-monotonica observada -> reframe mecanistico |
| Configuracao | unica (HUG-trained, todos features) | comparacao 2 regimes (HUG vs pex), 2 ablacoes (full-11, sem-NPS) |
| Mecanismo | implicito | explicito (extremes-overlap) |
| Via A2 | gate K1 absoluto | sensitivity high-value (NAO opcional) |
| AUC interno | 0.88 vendido como forca | 0.88 como sintoma do vies |
| Calibracao slope | co-primario | secundario (descritivo) |
| Modelo | logistic + PointNet++ | logistic apenas (PointNet++ adiado, nao essencial para tese) |

## EVIDENCIA EMPIRICA CONSOLIDADA

### M3-M4 baseline (HUG-trained, todos features)
- HUG MCA CV5 (n=118, 14R): AUC = 0.886 [0.819, 0.945]
- AneuX pex MCA (n=70, 30R): AUC = 0.588 [0.443, 0.723]
- CMHA MCA (n=105, 77R): AUC = 0.640 [0.523, 0.751]
- **Padrao non-monotonico** (CMHA > pex), sobrevive 3 ablacoes

### M4b bootstrap deltas
- Delta CMHA - pex (HUG-trained, full-11) = +0.052 [-0.130, +0.225]. CI cruza zero.
- Non-monotonicity formalmente nao significativa.

### M4c site shift audit (REFUTADO)
- HUG vs aneurisk+aneurist em nao-rompidos MCA: 0/11 features com KS p<0.05 & |d|>0.3.
- Site shift intra-AneuX nao explica o padrao.

### M4d distribuicao DENTRO de cada coorte
Cohen's d ruptured vs unruptured:
- HUG: AR +1.33, BF +1.52, EI +1.45, NSI +1.34 — **EXTREMO**
- Pex: AR +0.11, BF +0.33, EI +0.52, NSI +0.59 — fraco
- CMHA: AR +0.40, BF +0.08, EI +0.47, NSI +0.46 — fraco

Os 14 rompidos HUG MCA sao **morfologicamente extremos dentro da propria coorte**.

### M4e reverse-train (pex -> HUG + CMHA)
| Ablacao | pex-CV5 | pex->HUG | pex->CMHA |
|---|---|---|---|
| Full-11 | 0.657 | 0.670 | 0.592 |
| **C (sem NPS, 6f)** | 0.661 | **0.746** | **0.697** |

- pex-CV 0.66 vs HUG-CV 0.88 com CIs nao se tocam — **efeito interno robusto**
- Assimetria: pex->HUG 0.67 > HUG->pex 0.59 (rompidos HUG sao mais identificaveis;
  rompidos pex sao "medianos" e modelo HUG-trained nao reconhece)
- Sem features convencao-dependentes (NPS), pex-trained supera HUG-trained em ambos
  externos

### M4g extremes-overlap (mecanismo) — ablacao C
Spearman rho entre p_predicao e |z| extremidade dos rompidos CMHA:
- HUG-trained vs extrem_NPS: **rho = +0.434 (p=0.0001)**
- pex-trained vs extrem_NPS: **rho = -0.004 (p=0.97)**

AUC por tercil extremidade:
| | T1 moderado | T2 medio | T3 extremo | Gap T3-T1 |
|---|---|---|---|---|
| HUG-trained | 0.61 | 0.51 | 0.84 | +0.232 |
| pex-trained | 0.76 | 0.59 | 0.75 | -0.010 |

### M4h extremidade out-of-model (Q1 GPT — blindagem)
Extremidade calculada com NPS (excluidos do modelo C):
- HUG-trained vs Mahalanobis NPS: rho = +0.570 (p<0.0001)
- pex-trained vs Mahalanobis NPS: rho = +0.244 (p=0.032)

HUG-trained correlaciona com extremidade em features que **nem usa**. Refuta
circularidade trivial.

### M4i bootstrap interacao (Q2 GPT — pilar)
Bootstrap 1000 resamples, gap T3-T1 e diferenca de gaps:
- Gap HUG-trained: +0.234 [+0.099, +0.378]. **CI exclui zero. P(gap>0)=1.000.**
- Gap pex-trained: -0.009 [-0.180, +0.154]. CI inclui zero.
- **Interacao (gap_HUG - gap_pex): +0.244 [+0.079, +0.403]. CI exclui zero. P(diff>0)=0.998.**

Interacao modelo x tercil **e formalmente robusta**. Mecanismo extremes-overlap
e o pilar empirico, nao secondary.

## CONFIGURACAO DE ANALISE PRIMARIA v6.1

**Fatorial 2×2** (correcao GPT r4 — nao confundir regime com feature set):

| Celula | Regime | Ablacao | CV5 | cross_ext | CMHA |
|---|---|---|---|---|---|
| HUG × full-11 | HUG (14R/118) | 11 features | 0.886 | 0.588 | 0.640 |
| HUG × sem-NPS | HUG (14R/118) | 6 features | 0.882 | 0.627 | 0.653 |
| pex × full-11 | pex (30R/70) | 11 features | 0.657 | 0.670 | 0.592 |
| pex × sem-NPS | pex (30R/70) | 6 features | 0.661 | 0.746 | 0.697 |

**Efeito-regime** (media das ablacoes): Δ_CV5 = −0.225, Δ_cross = +0.101, **Δ_CMHA = 0.000**.
**Efeito-ablacao** (media dos regimes): Δ_cross = +0.057, Δ_CMHA = +0.059.
**Interacao regime×ablacao** em CMHA: pex beneficia 8× mais de remover NPS (+0.105 vs +0.013).

**Interpretacao**: efeito-regime e efeito-ablacao sao separaveis. O ganho pex+semNPS nao
e so "escolhi a celula favoravel" — e a combinacao de treino representativo (efeito regime)
com features convention-robust (efeito ablacao), ambos contribuindo aditivamente.

**Defesa contra revisor "modelo fraco":** pex-CV 0.66 nao e "modelo clinico"; e treino
mais transportavel. A COMPARACAO entre celulas e o produto, nao um modelo vencedor.

## METRICAS PRIMARIAS v6.1
1. **Grid fatorial 2×2**: AUC + 95% CI bootstrap por (regime × ablacao × dataset)
2. **Efeito-regime e efeito-ablacao** separados (margins do grid)
3. **Interacao modelo × tercil de extremidade** (M4i): pilar mecanistico
   - Gap T3−T1 HUG: +0.234 [+0.099, +0.378] (CI exclui zero)
   - Diff gap: +0.244 [+0.079, +0.403] (CI exclui zero)
4. Spearman rho predicao × extremidade out-of-model (M4h): blindagem causal
   - HUG-trained ρ=+0.57 (Mahalanobis NPS, p<0.0001), pex ρ=+0.24

## METRICAS SECUNDARIAS
- Calibration slope/intercept/Brier por celula
- Reliability plots por regime
- Sensitivity HUG1-only (M5e): confirma spectrum bias nao vem de snf
- Sensitivity excluindo pacientes mistos R+UR (M5c): sem impacto
- Sensitivity pooled HUG+pex (M5c): pooling piora (extremos HUG "contaminam")

## VIA A2 (CMHA visual) — STATUS v6.1 (ROTA A)
**v5:** gate K1 absoluto.
**v6.0:** sensitivity high-value, NAO gate.
**v6.1 (correcao GPT r4, Rota A):** **alta prioridade pre-submissao, NAO gate
mas NAO opcional.**

Razao: M4g/M4h/M4i (pilares mecanisticos) dependem de CMHA. Se manter como pilares,
Via A2 tem que estar resolvida. Sem A2, revisor aceita nucleo HUG-vs-pex mas rebaixa
CMHA a "supportive stress cohort with unresolved admissibility concerns" — isso
enfraquece a metade mecanistica do paper.

App Via A2 pronto (FastAPI+SQLite, 77 casos seeded). Aguarda ngrok + Noel.

## KILL CRITERIOS v6.1
| ID | Regra | Status |
|---|---|---|
| K1 | Via A2: se CMHA rompidos pos-procedimento > 80% → caveat severo. **Rota A: alta prioridade pre-submissao** | PENDENTE Noel |
| K2 | HUG denominador | RESOLVIDO |
| K3 | Shared core <10 features | RESOLVIDO (11) |
| K4 | >5% sem patient_id | RESOLVIDO (regex fallback p\d+) |
| K5 | PointNet++ | ADIADO (nao essencial para tese v6) |
| K6 | Ambos regimes colapsam (AUC <0.55 em ambos externos) | NAO disparado (pex>=0.59 em todas) |
| K7 | <50 AneuX MCA external | OK (n=70) |
| K8 | M4i interacao CI cruza zero | NAO disparado (CI exclui) |

## JOURNAL RANKING v6
**Inalterado:** EurRadiol / AJNR primario. Radiology:AI desce (paper agora menos
"AI-first", mais "metodologico-explicativo"). JNIS fora de fit (intervencionista).

## P(A1) v6.1
- Base case: **45-55%**
- Bom caso (Via A2 completa + fatorial bem comunicado + figuras fortes): **55-65%**
- Ruim (revisor le pex 0.66 como "fraco" e nao compra spectrum bias): **30-40%**

v6.1 nao muda P(A1) vs v6 — correcoes sao de robustez, nao de conteudo novo.

## EVIDENCIA COMPLEMENTAR (M5d + M5e)

### M5d — Fatorial 2×2 (correcao GPT r4 #1)
Grid completo permite separar contribuicoes:
- Efeito-regime puro em cross_ext: Δ = +0.101 (pex melhor, independente de ablacao)
- Efeito-ablacao puro: Δ = +0.06 (sem NPS melhor, independente de regime)
- Interacao: pex beneficia 8× mais de remover NPS (+0.105 vs +0.013)
- **No CMHA, efeito-regime e ZERO** (+0.002) quando mediado entre ablacoes.
  Isso importa: a vantagem pex→CMHA vem inteiramente da interacao com ablacao.

### M5e — HUG1-only sensitivity (correcao GPT r4 #2)
hug2016 sozinha (n=89, R=9): Cohen's d persiste extremo (AR +1.56, BF +1.74).
AUC → CMHA = 0.646 (vs 0.653 com snf). Delta −0.007: **spectrum bias NAO vem de snf**.
Spectrum bias e do fenotipo rompido HUG como um todo, nao de composicao subcoortal.

### M5c — Sensitivities adicionais
- Excluir pacientes mistos R+UR: sem impacto (5 pacientes MCA, delta ~0)
- Pooled HUG+pex → CMHA = 0.674 < pex-only 0.697: **extremos HUG "contaminam"
  o pooled**. Mais dados NAO ajudam quando positivos sao nao-representativos.

## TRANSPARENCIA / OSF v6.1
**NAO chamar de pre-registration em sentido forte.** v6.1 nasceu apos execucao M3-M5e.
Chamar de: **"locked analysis plan after exploratory phase"**.

Descricao honesta: "The original plan (v5) predicted monotonic cascade degradation.
Execution revealed non-monotonic AUC and led to the spectrum bias investigation.
The current analysis plan was locked after the exploratory findings but before
drafting the manuscript."

## CRONOGRAMA v6.1 (4-5 meses)
| Mes | Milestone |
|---|---|
| M1 [agora] | Memorial v6.1 congelar + Via A2 com Noel + figuras fatoriais |
| M2 | Draft Methods + Results |
| M3 | Draft completo + cover letter + EurRadiol pre-submission inquiry |
| M4 | Submissao EurRadiol |
| M5 | Cascade (se desk reject) — AJNR |

## DUVIDAS RESIDUAIS (para sessoes futuras)
1. Bootstrap CI do grid 2×2 inteiro (cada celula, cada dataset) — computar?
3. Tabela 1 deve incluir M4h Mahalanobis ou so M4g uni-mean(|z|)?
4. Pre-registration OSF v6: como descrever a pivotada de v5 honestamente sem queimar credibilidade?
5. Cover letter: vender como "we found the planned cascade was non-monotonic and the
   investigation became the paper" e honesto ou suspeito?

## ARQUIVOS
- Memorial v5 (superseded): memoria project_memorial_v5_2026_04_15.md
- Codigo M4-M4i: Y:/doutorado_noel/02_codigo/M4*.py
- Resultados M4f, M4h, M4i CSVs: Y:/doutorado_noel/03_analises/M4*.csv
- Briefing GPT r3: C:\Users\oncol\Desktop\briefing_m4fg_final_2026_04_15.md
- Resposta GPT r3: C:\Users\oncol\Desktop\gpt_response.md
