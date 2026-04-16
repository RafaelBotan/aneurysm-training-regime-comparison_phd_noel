"""
Via A2 — App de revisao radiologica para K1 do estudo CMHA.

Interface web que apresenta, um por um, os 77 aneurismas rompidos do
CMHA e coleta do Dr. Noel o veredicto: CTA adquirida antes ou depois
do sangramento.

Execucao local:
    cd Y:/doutorado_noel/02_codigo/via_a2_app
    python -m uvicorn app:app --host 0.0.0.0 --port 8095 --reload

Tunel externo (ngrok):
    python ngrok_tunnel.py
"""
from __future__ import annotations

import csv
import io
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -------------------------------------------------------------------
# Caminhos
# -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent  # Y:/doutorado_noel
BRIEFING_DIR = PROJECT_ROOT / "00_briefing"
IMG_DIR = BRIEFING_DIR / "via_A2_screenshots"
TEMPLATE_CSV = BRIEFING_DIR / "cmha_via_A2_template.csv"
DB_PATH = APP_DIR / "via_a2.db"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

# -------------------------------------------------------------------
# Dominio
# -------------------------------------------------------------------
PROXIES = [
    ("sah_cisternal",  "Sangue nas cisternas basais / fissura de Sylvius"),
    ("ivh",            "Sangue nos ventrículos (hemorragia intraventricular)"),
    ("hematoma",       "Hematoma no parênquima adjacente ao aneurisma"),
    ("hydrocephalus",  "Hidrocefalia aguda (ventrículos dilatados)"),
    ("evd",            "Dreno ventricular externo"),
    ("clip",           "Clipe metálico (pós-clipagem)"),
    ("coil",           "Coil / espiral endovascular"),
    ("edema",          "Edema cerebral difuso / apagamento sulcal"),
    ("craniotomy",     "Craniotomia ou craniectomia (defeito ósseo)"),
]
PROXY_KEYS = [k for k, _ in PROXIES]

VERDICT_LABELS = {
    "LIKELY_PRE":    "Antes do sangramento",
    "LIKELY_POST":   "Depois do sangramento",
    "INDETERMINATE": "Indeterminado",
}

# -------------------------------------------------------------------
# Inicializacao de DB
# -------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id    TEXT UNIQUE NOT NULL,
    gender        TEXT,
    age           REAL,
    location      TEXT,
    image_file    TEXT NOT NULL,
    verdict       TEXT,
    sah_cisternal INTEGER DEFAULT 0,
    ivh           INTEGER DEFAULT 0,
    hematoma      INTEGER DEFAULT 0,
    hydrocephalus INTEGER DEFAULT 0,
    evd           INTEGER DEFAULT 0,
    clip          INTEGER DEFAULT 0,
    coil          INTEGER DEFAULT 0,
    edema         INTEGER DEFAULT 0,
    craniotomy    INTEGER DEFAULT 0,
    notes         TEXT,
    completed_at  TEXT,
    updated_at    TEXT
);
"""


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def seed_if_needed() -> None:
    """Cria o schema e popula a partir do template CSV + PNGs disponiveis."""
    with closing(db_connect()) as conn:
        conn.executescript(SCHEMA)
        count = conn.execute("SELECT COUNT(*) AS c FROM cases").fetchone()["c"]
        if count > 0:
            return

        # Carregar template CSV (pacientes + metadados)
        template = pd.read_csv(TEMPLATE_CSV)

        # Limitar aos pacientes com PNG disponivel
        available = {p.stem for p in IMG_DIR.glob("*.png")}

        inserted = 0
        for _, row in template.iterrows():
            pid = str(row["number"]).strip()
            if pid not in available:
                continue
            conn.execute(
                """
                INSERT INTO cases
                  (patient_id, gender, age, location, image_file)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    row.get("Gender"),
                    row.get("Age"),
                    row.get("location"),
                    f"{pid}.png",
                ),
            )
            inserted += 1
        conn.commit()
        print(f"[seed] inseridos {inserted} casos.")


# -------------------------------------------------------------------
# FastAPI
# -------------------------------------------------------------------
app = FastAPI(title="Via A2 — Revisao radiologica CMHA")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/img", StaticFiles(directory=IMG_DIR), name="img")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.on_event("startup")
def _startup() -> None:
    STATIC_DIR.mkdir(exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
    seed_if_needed()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_progress() -> dict:
    with closing(db_connect()) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM cases").fetchone()["c"]
        done = conn.execute(
            "SELECT COUNT(*) c FROM cases WHERE verdict IS NOT NULL"
        ).fetchone()["c"]
        pre = conn.execute(
            "SELECT COUNT(*) c FROM cases WHERE verdict='LIKELY_PRE'"
        ).fetchone()["c"]
        post = conn.execute(
            "SELECT COUNT(*) c FROM cases WHERE verdict='LIKELY_POST'"
        ).fetchone()["c"]
        indet = conn.execute(
            "SELECT COUNT(*) c FROM cases WHERE verdict='INDETERMINATE'"
        ).fetchone()["c"]
    pct = int(round(100 * done / total)) if total else 0
    return {
        "total": total, "done": done, "remaining": total - done,
        "pre": pre, "post": post, "indet": indet,
        "pct": pct,
    }


def fetch_case(case_id: int) -> Optional[sqlite3.Row]:
    with closing(db_connect()) as conn:
        return conn.execute(
            "SELECT * FROM cases WHERE id=?", (case_id,)
        ).fetchone()


def next_unreviewed_id() -> Optional[int]:
    with closing(db_connect()) as conn:
        row = conn.execute(
            "SELECT id FROM cases WHERE verdict IS NULL ORDER BY id LIMIT 1"
        ).fetchone()
        return row["id"] if row else None


# -------------------------------------------------------------------
# Rotas
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    prog = get_progress()
    next_id = next_unreviewed_id()
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "progress": prog,
            "next_id": next_id,
        },
    )


@app.get("/tutorial", response_class=HTMLResponse)
def tutorial(request: Request):
    return templates.TemplateResponse(
        "tutorial.html",
        {"request": request, "proxies": PROXIES},
    )


@app.get("/review", response_class=HTMLResponse)
def review_root():
    nid = next_unreviewed_id()
    if nid is None:
        return RedirectResponse(url="/done", status_code=302)
    return RedirectResponse(url=f"/review/{nid}", status_code=302)


@app.get("/review/{case_id}", response_class=HTMLResponse)
def review_case(request: Request, case_id: int):
    case = fetch_case(case_id)
    if case is None:
        raise HTTPException(404, "Caso não encontrado")

    # Posicao na lista ordenada
    with closing(db_connect()) as conn:
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM cases ORDER BY id").fetchall()]
    try:
        idx = ids.index(case_id)
    except ValueError:
        idx = 0
    prev_id = ids[idx - 1] if idx > 0 else None
    next_id = ids[idx + 1] if idx < len(ids) - 1 else None

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "case": dict(case),
            "proxies": PROXIES,
            "verdict_labels": VERDICT_LABELS,
            "progress": get_progress(),
            "pos": idx + 1,
            "total": len(ids),
            "prev_id": prev_id,
            "next_id": next_id,
        },
    )


@app.post("/review/{case_id}")
async def save_case(
    case_id: int,
    request: Request,
    verdict: str = Form(...),
    notes: str = Form(""),
    action: str = Form("next"),
):
    form = await request.form()
    proxy_values = {k: (1 if form.get(k) else 0) for k in PROXY_KEYS}

    if verdict not in VERDICT_LABELS:
        raise HTTPException(400, "Veredicto invalido")

    now = datetime.now().isoformat(timespec="seconds")
    with closing(db_connect()) as conn:
        conn.execute(
            f"""
            UPDATE cases SET
              verdict=?, notes=?,
              {', '.join(f'{k}=?' for k in PROXY_KEYS)},
              completed_at=COALESCE(completed_at, ?),
              updated_at=?
            WHERE id=?
            """,
            (
                verdict,
                notes.strip(),
                *[proxy_values[k] for k in PROXY_KEYS],
                now,
                now,
                case_id,
            ),
        )
        conn.commit()

    # Proximo destino
    if action == "prev":
        with closing(db_connect()) as conn:
            row = conn.execute(
                "SELECT id FROM cases WHERE id<? ORDER BY id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row:
            return RedirectResponse(url=f"/review/{row['id']}", status_code=303)
        return RedirectResponse(url="/", status_code=303)

    if action == "stay":
        return RedirectResponse(url=f"/review/{case_id}", status_code=303)

    # default = next
    with closing(db_connect()) as conn:
        row = conn.execute(
            "SELECT id FROM cases WHERE id>? ORDER BY id LIMIT 1",
            (case_id,),
        ).fetchone()
    if row:
        return RedirectResponse(url=f"/review/{row['id']}", status_code=303)

    nid = next_unreviewed_id()
    if nid:
        return RedirectResponse(url=f"/review/{nid}", status_code=303)
    return RedirectResponse(url="/done", status_code=303)


@app.get("/done", response_class=HTMLResponse)
def done(request: Request):
    return templates.TemplateResponse(
        "done.html",
        {"request": request, "progress": get_progress()},
    )


@app.get("/export")
def export_csv():
    with closing(db_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM cases ORDER BY patient_id"
        ).fetchall()

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = [
        "patient_id", "gender", "age", "location", "verdict",
        *PROXY_KEYS, "notes", "completed_at", "updated_at",
    ]
    writer.writerow(header)
    for r in rows:
        writer.writerow([r[h] for h in header])

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_data = buf.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="via_A2_verdicts_{ts}.csv"'
        },
    )


@app.get("/reset", response_class=HTMLResponse)
def reset_confirm(request: Request):
    return templates.TemplateResponse(
        "reset.html",
        {"request": request, "progress": get_progress()},
    )


@app.post("/reset")
def reset_do():
    with closing(db_connect()) as conn:
        conn.execute(
            f"""
            UPDATE cases SET
              verdict=NULL, notes=NULL,
              {', '.join(f'{k}=0' for k in PROXY_KEYS)},
              completed_at=NULL, updated_at=NULL
            """
        )
        conn.commit()
    return RedirectResponse(url="/", status_code=303)
