"""Misure di rendimento e rischio su serie mensili, e bootstrap stazionario per la significatività."""

from __future__ import annotations

import numpy as np
import pandas as pd


def crescita(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def drawdown(r: pd.Series) -> pd.Series:
    c = crescita(r)
    return c / c.cummax().clip(lower=1.0) - 1


def cagr(r: pd.Series, periodi_anno: int = 12) -> float:
    return float((1 + r).prod() ** (periodi_anno / len(r)) - 1)


def sharpe(r: pd.Series, rf: pd.Series, periodi_anno: int = 12) -> float:
    e = r - rf
    s = e.std()
    return float(e.mean() / s * np.sqrt(periodi_anno)) if s > 0 else float("nan")


def metriche(r: pd.Series, rf: pd.Series, benchmark: pd.Series | None = None, periodi_anno: int = 12) -> dict:
    """Rendimento composto, volatilità, Sharpe, Sortino, massimo drawdown, Calmar, CVaR e misure relative."""
    e = r - rf
    giu = np.sqrt((np.minimum(e, 0) ** 2).mean())
    dd = drawdown(r)
    q = r.quantile(0.05)
    out = {
        "Rendimento annuo": cagr(r, periodi_anno),
        "Volatilità": float(r.std() * np.sqrt(periodi_anno)),
        "Sharpe": sharpe(r, rf, periodi_anno),
        "Sortino": float(e.mean() / giu * np.sqrt(periodi_anno)) if giu > 0 else float("nan"),
        "Massimo drawdown": float(dd.min()),
        "Calmar": cagr(r, periodi_anno) / abs(dd.min()) if dd.min() < 0 else float("nan"),
        "CVaR 95% mensile": float(r[r <= q].mean()),
        "Mese peggiore": float(r.min()),
        "Mesi positivi": float((r > 0).mean()),
    }
    if benchmark is not None:
        a = r - benchmark
        te = a.std() * np.sqrt(periodi_anno)
        out["Extra-rendimento"] = cagr(r, periodi_anno) - cagr(benchmark, periodi_anno)
        out["Tracking error"] = float(te)
        out["Information ratio"] = float(a.mean() * periodi_anno / te) if te > 0 else float("nan")
    return out


def indici_bootstrap(n: int, lunghezza_media: float, rng: np.random.Generator) -> np.ndarray:
    """Un campione del bootstrap stazionario di Politis e Romano (1994): blocchi di lunghezza geometrica."""
    idx = np.empty(n, dtype=int)
    p = 1.0 / lunghezza_media
    idx[0] = rng.integers(n)
    nuovi = rng.random(n) < p
    casuali = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = casuali[t] if nuovi[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_differenza(r: pd.Series, b: pd.Series, rf: pd.Series, n: int = 2000, lunghezza_media: float = 6.0,
                         seed: int = 0) -> pd.DataFrame:
    """Distribuzione bootstrap della differenza di rendimento annuo e di Sharpe tra due strategie.

    I blocchi conservano l'autocorrelazione e la persistenza dei regimi; strategia e benchmark si ricampionano
    sugli stessi mesi.
    """
    rng = np.random.default_rng(seed)
    R, B, F = r.values, b.values, rf.values
    out = np.empty((n, 2))
    for i in range(n):
        j = indici_bootstrap(len(R), lunghezza_media, rng)
        rr, bb, ff = R[j], B[j], F[j]
        out[i, 0] = np.prod(1 + rr) ** (12 / len(rr)) - np.prod(1 + bb) ** (12 / len(bb))
        sr = (rr - ff).mean() / (rr - ff).std(ddof=1)
        sb = (bb - ff).mean() / (bb - ff).std(ddof=1)
        out[i, 1] = (sr - sb) * np.sqrt(12)
    return pd.DataFrame(out, columns=["Extra-rendimento", "Differenza di Sharpe"])
