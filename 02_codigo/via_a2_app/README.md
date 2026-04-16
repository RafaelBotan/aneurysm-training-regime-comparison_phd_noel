# Via A2 — App de revisão radiológica

Aplicação web para o Dr. Noel classificar as 77 tomografias do CMHA
como **antes** ou **depois** do sangramento.

## Estrutura

```
via_a2_app/
├── app.py              # backend FastAPI
├── ngrok_tunnel.py     # tunel externo opcional
├── run.ps1             # launcher Windows PowerShell
├── requirements.txt
├── static/style.css
├── templates/*.html
└── via_a2.db           # criado na primeira execucao
```

As imagens são servidas diretamente de
`Y:/doutorado_noel/00_briefing/via_A2_screenshots/`.

## Como rodar (local)

```powershell
cd Y:\doutorado_noel\02_codigo\via_a2_app
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8095 --reload
# abrir http://localhost:8095
```

Ou via script:

```powershell
.\run.ps1             # apenas local
.\run.ps1 -Ngrok      # local + tunel ngrok
```

## Como expor via ngrok

Requer `NGROK_AUTHTOKEN` (pegar em https://dashboard.ngrok.com/get-started/your-authtoken).

```powershell
$env:NGROK_AUTHTOKEN="seu_token_aqui"
python ngrok_tunnel.py --port 8095
```

## Exportar resultados

O botão **Baixar CSV** (na topbar ou na página "Concluída") exporta
todos os veredictos no formato esperado pelo pipeline do estudo.

## Resetar

Página `/reset` apaga todos os veredictos (pede confirmação).

## Atalhos de teclado na revisão

- `1` → Antes do sangramento
- `2` → Depois do sangramento
- `3` → Indeterminado
- `Ctrl+Enter` → Salvar e próximo
