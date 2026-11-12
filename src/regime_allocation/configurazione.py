"""Configurazione: esempio dimostrativo e lettura da file JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import dati
from .analisi import Configurazione
from .backtest import Parametri

CARATTERISTICHE = ["Azioni globali", "Titoli di Stato euro", "Obbligazioni societarie"]
RISCHIOSI = ["Azioni globali", "Obbligazioni societarie"]
RIFUGI = ["Titoli di Stato euro", "Liquidità"]
BANDE = {"Inflazione euro": 0.05, "Oro": 0.05}  # le attività con peso piccolo si muovono meno


def esempio(cliente: str = "Famiglia Conti", capitale: float = 600_000.0) -> Configurazione:
    par = Parametri(caratteristiche=list(CARATTERISTICHE), liquidita=dati.LIQUIDITA, rischiosi=list(RISCHIOSI),
                    rifugi=list(RIFUGI), bande=dict(BANDE))
    return Configurazione(benchmark=dati.BENCHMARK.copy(), parametri=par, cliente=cliente, capitale=capitale)


def _frazione(d: dict, chiave: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    x = float(d.get(chiave, default))
    if not lo <= x <= hi:
        raise ValueError(f"'{chiave}' deve essere tra {lo} e {hi} (ora {x})")
    return x


def da_dizionario(d: dict, base: Path | None = None) -> Configurazione:
    """Costruisce la configurazione; i percorsi sono relativi a ``base``."""
    if "benchmark" not in d or not d["benchmark"]:
        raise ValueError("Manca il benchmark: indicare i pesi strategici in 'benchmark'")
    bench = pd.Series({str(k): float(v) for k, v in d["benchmark"].items()})
    if (bench < 0).any():
        raise ValueError("I pesi del benchmark non possono essere negativi")
    if abs(bench.sum() - 1) > 1e-6:
        raise ValueError(f"I pesi del benchmark devono sommare a 1 (ora {bench.sum():.4f})")
    liquidita = d.get("liquidita", dati.LIQUIDITA)
    car = list(d.get("caratteristiche", CARATTERISTICHE))
    if len(car) < 1:
        raise ValueError("Indicare almeno una attività in 'caratteristiche'")
    k = int(d.get("regimi", 3))
    if not 2 <= k <= 5:
        raise ValueError("'regimi' deve essere tra 2 e 5")
    par = Parametri(
        caratteristiche=car, liquidita=liquidita, n_regimi=k,
        persistenza=_frazione(d, "persistenza", 0.0, 0, 1e6), inizio_test=str(d.get("inizio_test", "2007-01")),
        rifai_ogni=int(d.get("rifai_ogni", 12)), banda=_frazione(d, "banda", 0.10),
        bande={str(a): _frazione(d["bande"], a, 0.0) for a in d.get("bande", {})},
        orizzonte_costi=_frazione(d, "orizzonte_costi", 3.0, 1.0, 120.0),
        avversione=_frazione(d, "avversione", 5.0, 0.0, 100.0), costo=_frazione(d, "costo", 0.0010, 0, 0.05),
        restringimento=_frazione(d, "restringimento", 0.5), soglia_crisi=_frazione(d, "soglia_crisi", 0.5),
        rischiosi=list(d.get("rischiosi", [a for a in RISCHIOSI if a in bench.index])),
        rifugi=list(d.get("rifugi", [a for a in RIFUGI if a in bench.index])),
    )
    if par.rifai_ogni < 1:
        raise ValueError("'rifai_ogni' deve essere almeno 1 mese")
    if not par.rifugi:
        raise ValueError("Indicare almeno un rifugio per la regola risk-off in 'rifugi'")
    rend = d.get("rendimenti")
    if rend and base is not None:
        rend = str((base / rend).resolve())
    episodi = [(e["nome"], e["inizio"], e["fine"]) for e in d["episodi"]] if "episodi" in d else list(dati.EPISODI)
    capitale = float(d.get("capitale", 500_000))
    if capitale <= 0:
        raise ValueError("Il capitale deve essere positivo")
    return Configurazione(benchmark=bench, parametri=par, cliente=d.get("cliente", "Cliente"), capitale=capitale,
                          rendimenti=rend, episodi=episodi, confronto_regimi=bool(d.get("confronto_regimi", True)),
                          n_bootstrap=int(d.get("bootstrap", 2000)))


def carica(percorso: str | Path) -> Configurazione:
    percorso = Path(percorso)
    return da_dizionario(json.loads(percorso.read_text(encoding="utf-8")), percorso.parent)
