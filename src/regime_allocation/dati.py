"""Dati dimostrativi: rendimenti settimanali simulati di sei classi di attività dal 2000, con regimi noti.

La storia ricalca per date e ordine di grandezza quella dei mercati in euro: la bolla tecnologica, la crisi
finanziaria del 2008, il debito sovrano del 2011, la pandemia, il rialzo dei tassi del 2022. Ogni mese ha un
regime «vero» (Espansione, Tassi in rialzo, Crisi) fissato dal calendario degli episodi; dato il regime, i
rendimenti settimanali in eccesso sulla liquidità sono estratti da una t di Student multivariata (code più spesse
della normale) con medie, volatilità e correlazioni proprie del regime. Il modello HMM non conosce né il
calendario né la distribuzione: deve ritrovarli dai soli rendimenti. Conoscere i regimi veri permette di
misurare quanto bene e con quanto ritardo il modello li riconosce, cosa impossibile con dati reali.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ATTIVI = ["Azioni globali", "Titoli di Stato euro", "Obbligazioni societarie", "Inflazione euro", "Oro", "Liquidità"]
LIQUIDITA = "Liquidità"
REGIMI = ["Espansione", "Tassi in rialzo", "Crisi"]
INIZIO, FINE = "2000-01", "2026-10"

# Benchmark strategico di un profilo moderato
BENCHMARK = pd.Series({"Azioni globali": 0.40, "Titoli di Stato euro": 0.25, "Obbligazioni societarie": 0.15,
                       "Inflazione euro": 0.05, "Oro": 0.05, "Liquidità": 0.10})

# Calendario dei regimi (mesi inclusi); i mesi non elencati sono di espansione
EPISODI_VERI = {
    "Crisi": [("2000-09", "2001-04"), ("2001-08", "2001-10"), ("2002-05", "2003-03"), ("2007-11", "2009-03"),
              ("2010-05", "2010-06"), ("2011-07", "2012-05"), ("2015-08", "2015-09"), ("2016-01", "2016-02"),
              ("2018-10", "2018-12"), ("2020-02", "2020-04"), ("2025-03", "2025-04")],
    "Tassi in rialzo": [("2005-12", "2006-06"), ("2013-05", "2013-08"), ("2015-04", "2015-06"),
                        ("2018-02", "2018-03"), ("2021-12", "2022-10"), ("2023-08", "2023-10")],
}

# Episodi da analizzare nel backtest
EPISODI = [
    ("Crisi finanziaria 2008", "2007-11", "2009-03"),
    ("Debito sovrano 2011", "2011-07", "2012-05"),
    ("Correzione di fine 2018", "2018-10", "2018-12"),
    ("Pandemia 2020", "2020-02", "2020-04"),
    ("Inflazione 2022", "2021-12", "2022-10"),
    ("Dazi 2025", "2025-03", "2025-04"),
]

# Tasso della liquidità (Euribor 3 mesi, valori medi annui approssimati)
_TASSO = {2000: 4.4, 2001: 4.3, 2002: 3.3, 2003: 2.3, 2004: 2.1, 2005: 2.2, 2006: 3.1, 2007: 4.3, 2008: 4.6,
          2009: 1.2, 2010: 0.8, 2011: 1.4, 2012: 0.6, 2013: 0.2, 2014: 0.2, 2015: 0.0, 2016: -0.3, 2017: -0.33,
          2018: -0.32, 2019: -0.36, 2020: -0.43, 2021: -0.55, 2022: 0.35, 2023: 3.4, 2024: 3.6, 2025: 2.2,
          2026: 2.0}

# Rendimenti in eccesso annui e volatilità annue per regime: azioni, Stato, societarie, inflazione, oro
_MEDIE = {"Espansione": [0.12, 0.015, 0.035, 0.02, 0.05],
          "Tassi in rialzo": [-0.06, -0.12, -0.10, -0.04, -0.02],
          "Crisi": [-0.20, 0.05, -0.04, -0.01, 0.10]}
_VOL = {"Espansione": [0.12, 0.04, 0.035, 0.045, 0.15],
        "Tassi in rialzo": [0.17, 0.07, 0.065, 0.06, 0.16],
        "Crisi": [0.24, 0.06, 0.07, 0.07, 0.22]}
# Correlazioni nell'ordine (az-St, az-soc, az-inf, az-oro, St-soc, St-inf, St-oro, soc-inf, soc-oro, inf-oro)
_CORR = {"Espansione": [-0.20, 0.30, 0.10, 0.05, 0.70, 0.60, 0.15, 0.50, 0.10, 0.25],
         "Tassi in rialzo": [0.55, 0.60, 0.40, 0.20, 0.85, 0.70, 0.30, 0.65, 0.25, 0.40],
         "Crisi": [-0.45, 0.55, 0.20, -0.10, 0.20, 0.50, 0.30, 0.40, 0.00, 0.30]}
GRADI_LIBERTA = 10


def _matrice_corr(v: list[float]) -> np.ndarray:
    C = np.eye(5)
    C[np.triu_indices(5, 1)] = v
    return np.triu(C) + np.triu(C, 1).T


def settimane() -> pd.DatetimeIndex:
    """Venerdì del campione dimostrativo."""
    return pd.date_range(f"{INIZIO}-01", pd.Period(FINE, "M").to_timestamp(how="end").normalize(), freq="W-FRI")


def regimi_veri(indice: pd.DatetimeIndex | None = None) -> pd.Series:
    """Regime vero di ogni periodo (settimana o mese) del campione dimostrativo."""
    if indice is None:
        indice = settimane()
    s = pd.Series("Espansione", index=indice, name="Regime")
    per = indice.to_period("M")
    for nome, ep in EPISODI_VERI.items():
        for a, b in ep:
            s[(per >= pd.Period(a, "M")) & (per <= pd.Period(b, "M"))] = nome
    return s


def tasso_liquidita(indice: pd.DatetimeIndex, periodi_anno: int = 52) -> pd.Series:
    """Rendimento della liquidità per periodo, interpolato tra le medie annue."""
    anni = np.array(sorted(_TASSO)) + 0.5
    x = indice.year + (indice.day_of_year - 0.5) / 365.25
    return pd.Series(np.interp(x, anni, [_TASSO[a] / 100 for a in sorted(_TASSO)]) / periodi_anno, index=indice)


def rendimenti_dimostrativi(seed: int = 2026) -> pd.DataFrame:
    """Rendimenti settimanali totali simulati (decimali) delle sei classi di attività."""
    regime = regimi_veri()
    idx = regime.index
    rng = np.random.default_rng(seed)
    rf = tasso_liquidita(idx)
    eccesso = np.empty((len(idx), 5))
    nu = GRADI_LIBERTA
    for nome in REGIMI:
        m = regime.values == nome
        n = int(m.sum())
        vol = np.array(_VOL[nome]) / np.sqrt(52)
        cov = _matrice_corr(_CORR[nome]) * np.outer(vol, vol)
        z = rng.standard_normal((n, 5)) @ np.linalg.cholesky(cov).T
        scala = np.sqrt(rng.chisquare(nu, n) / (nu - 2))[:, None]  # t di Student con la stessa varianza
        eccesso[m] = np.array(_MEDIE[nome]) / 52 + z / scala
    r = pd.DataFrame(eccesso, index=idx, columns=ATTIVI[:-1]).add(rf, axis=0)
    r[LIQUIDITA] = rf
    r.index.name = "data"
    return r


def periodi_per_anno(indice: pd.DatetimeIndex) -> int:
    """52 per dati settimanali, 12 per dati mensili (dalla distanza mediana tra le date)."""
    giorni = np.median(np.diff(indice.values).astype("timedelta64[D]").astype(float))
    if giorni <= 10:
        return 52
    if 25 <= giorni <= 35:
        return 12
    raise ValueError(f"Frequenza non riconosciuta (distanza mediana {giorni:.0f} giorni): servono dati settimanali "
                     "o mensili")


def mensili(rendimenti: pd.DataFrame) -> pd.DataFrame:
    """Rendimenti mensili composti, indicizzati all'ultimo giorno del mese."""
    m = (1 + rendimenti).groupby(rendimenti.index.to_period("M")).prod() - 1
    m.index = m.index.to_timestamp(how="end").normalize()
    m.index.name = "data"
    return m


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def salva_csv(rendimenti: pd.DataFrame, percorso: str | Path, regimi: pd.Series | None = None) -> Path:
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    df = rendimenti.copy()
    if regimi is not None:
        df["regime"] = regimi
    df.index = df.index.strftime("%Y-%m-%d")
    df.index.name = "data"
    df.to_csv(percorso, float_format="%.6f")
    return percorso


def carica_csv(percorso: str | Path) -> tuple[pd.DataFrame, pd.Series | None]:
    """Legge rendimenti settimanali o mensili (colonna ``data``, una colonna per attività, decimali).

    Una colonna facoltativa ``regime`` con i regimi veri (solo per dati simulati) abilita le misure di
    riconoscimento. Se i valori sembrano in percentuale (|r| > 1 frequente) vengono divisi per 100.
    """
    df = pd.read_csv(percorso)
    if "data" not in df.columns:
        raise ValueError("Il CSV deve avere una colonna 'data'")
    df = df.sort_values("data", kind="stable").reset_index(drop=True)
    d = df.pop("data").astype(str)
    if d.str.len().max() <= 7:  # AAAA-MM: fine del mese
        df.index = pd.PeriodIndex(d, freq="M").to_timestamp(how="end").normalize()
    else:
        df.index = pd.to_datetime(d)
    df.index.name = "data"
    regimi = df.pop("regime") if "regime" in df.columns else None
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Valori mancanti o non numerici nei rendimenti: " +
                         ", ".join(df.columns[df.isna().any()]))
    if (df.abs() > 1).mean().mean() > 0.05:
        df = df / 100
    anni = (df.index[-1] - df.index[0]).days / 365.25
    if anni < 5:
        raise ValueError(f"Servono almeno 5 anni di dati, ce ne sono {anni:.1f}")
    periodi_per_anno(df.index)
    if regimi is not None:
        regimi.index = df.index
        regimi.name = "Regime"
    return df, regimi
