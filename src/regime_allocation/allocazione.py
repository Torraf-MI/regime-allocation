"""Allocazione tattica: momenti attesi dalle probabilità dei regimi e ottimizzazione entro bande dal benchmark.

Se nel mese prossimo il mercato sarà nel regime k con probabilità p_k, i rendimenti in eccesso hanno una
distribuzione mistura con

    μ = Σ p_k μ_k        Σ = Σ p_k (Σ_k + μ_k μ_kᵀ) − μ μᵀ

(il secondo termine aggiunge l'incertezza sul regime stesso). Il portafoglio massimizza
μᵀw − (λ/2) wᵀΣw − c·(12/h)·Σ|w − w₀| con pesi entro ± una banda dal benchmark strategico: è la logica di una
allocazione tattica (TAA) intorno a un'allocazione strategica (SAA) fissata dal mandato. Il costo di
transazione si paga una volta, mentre μ e Σ sono annui: lo si ammortizza su un orizzonte di h mesi, la durata
attesa di una posizione tattica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def momenti_misti(probabilita: np.ndarray, medie: np.ndarray, covarianze: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media e covarianza della mistura dei regimi."""
    p = np.asarray(probabilita, float)
    mu = p @ medie
    S = np.einsum("k,kij->ij", p, covarianze + np.einsum("ki,kj->kij", medie, medie)) - np.outer(mu, mu)
    return mu, 0.5 * (S + S.T)


def rendimenti_equilibrio(S: np.ndarray, pesi: np.ndarray, avversione: float) -> np.ndarray:
    """Rendimenti in eccesso impliciti nel benchmark: quelli per cui il benchmark è ottimo (π = λΣw)."""
    return avversione * S @ pesi


def ancora(medie: np.ndarray, media_totale: np.ndarray, equilibrio: np.ndarray, restringimento: float) -> np.ndarray:
    """Rendimenti attesi per regime ancorati all'equilibrio: μₖ = π + (1 − r)(μ̂ₖ − μ̂).

    La media storica μ̂ dipende molto dal campione (dopo il 2000-2003 e il 2008 le azioni avevano rendimenti medi
    negativi) e spingerebbe il portafoglio lontano dal benchmark per ragioni che non hanno a che fare con i regimi.
    Dei regimi si usa quindi solo lo scarto dalla media storica, ridotto di una quota r per l'errore di stima, e
    lo si somma ai rendimenti di equilibrio impliciti nel benchmark: con probabilità pari a quelle di lungo
    periodo il portafoglio ottimo resta vicino al benchmark, e se ne allontana solo quando un regime è più o meno
    probabile del solito.
    """
    return equilibrio + (1 - restringimento) * (medie - media_totale)


def limiti(riferimento: np.ndarray, banda: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.clip(riferimento - banda, 0, 1), np.clip(riferimento + banda, 0, 1)


def ottimizza(mu: np.ndarray, S: np.ndarray, riferimento: np.ndarray, precedente: np.ndarray | None = None,
              avversione: float = 5.0, banda: float | np.ndarray = 0.10, costo: float = 0.0,
              orizzonte: float = 3.0) -> np.ndarray:
    """Pesi ottimi media-varianza entro le bande, con costi di transazione proporzionali.

    Variabili: pesi w, acquisti b ≥ 0 e vendite s ≥ 0 con w = w₀ + b − s, così il costo è lineare e il
    problema resta convesso (SLSQP con gradienti analitici).
    """
    n = len(mu)
    rif = np.asarray(riferimento, float)
    w0 = rif.copy() if precedente is None else np.asarray(precedente, float)
    lo, hi = limiti(rif, banda)
    k = 12 / orizzonte * costo

    def f(x):
        w, b, s = x[:n], x[n:2 * n], x[2 * n:]
        Sw = S @ w
        val = -(mu @ w) + 0.5 * avversione * w @ Sw + k * (b.sum() + s.sum())
        g = np.concatenate([-mu + avversione * Sw, np.full(n, k), np.full(n, k)])
        return val, g

    A_eq = np.zeros((n + 1, 3 * n))
    A_eq[:n, :n], A_eq[:n, n:2 * n], A_eq[:n, 2 * n:] = np.eye(n), -np.eye(n), np.eye(n)
    A_eq[n, :n] = 1.0
    b_eq = np.r_[w0, 1.0]
    vincoli = [{"type": "eq", "fun": lambda x: A_eq @ x - b_eq, "jac": lambda x: A_eq}]
    x0w = np.clip(w0, lo, hi)
    x0 = np.concatenate([x0w, np.clip(x0w - w0, 0, None), np.clip(w0 - x0w, 0, None)])
    bounds = list(zip(lo, hi)) + [(0, 1)] * (2 * n)
    r = minimize(f, x0, jac=True, method="SLSQP", bounds=bounds, constraints=vincoli,
                 options={"maxiter": 300, "ftol": 1e-11})
    w = np.clip(r.x[:n], lo, hi)
    return w / w.sum()


def regola_risk_off(riferimento: pd.Series, p_crisi: float, soglia: float, banda: float | pd.Series,
                    rischiosi: list[str], rifugi: list[str]) -> pd.Series:
    """Regola semplice: se la probabilità di crisi supera la soglia, gli attivi rischiosi scendono al limite
    inferiore della banda e il ricavato va ai rifugi (fino al loro limite superiore, il resto alla liquidità)."""
    w = riferimento.copy()
    if p_crisi <= soglia:
        return w
    banda = banda if isinstance(banda, pd.Series) else pd.Series(banda, index=riferimento.index)
    liberato = 0.0
    for a in rischiosi:
        d = min(w[a], banda[a])
        w[a] -= d
        liberato += d
    for a in rifugi:
        d = min(liberato / len(rifugi), max(0.0, riferimento[a] + banda[a] - w[a]))
        w[a] += d
    resto = riferimento.sum() - w.sum()
    w[rifugi[-1]] += resto
    return w
