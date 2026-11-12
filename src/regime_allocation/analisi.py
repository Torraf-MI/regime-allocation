"""Analisi completa: regimi stimati, backtest fuori campione, confronti, robustezza e proposta per il cliente."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import dati
from .allocazione import ancora, momenti_misti, ottimizza, regola_risk_off, rendimenti_equilibrio
from .backtest import Parametri, Previsioni, Risultato, senno_di_poi, simula, walk_forward
from .hmm import selezione_regimi
from .metriche import bootstrap_differenza, metriche
from .regimi import StimaRegimi, stima_regimi

BENCH, REGIMI, MV, RISKOFF, INGENUO = ("Benchmark statico", "Allocazione per regimi", "Media-varianza senza regimi",
                                       "Regola risk-off", "Backtest ingenuo (senno di poi)")
STRATEGIE = [BENCH, REGIMI, MV, RISKOFF, INGENUO]
BANDE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
COSTI = [0.0, 0.0010, 0.0025, 0.0050, 0.0100]


@dataclass
class Configurazione:
    benchmark: pd.Series
    parametri: Parametri
    cliente: str = "Cliente"
    capitale: float = 500_000.0
    rendimenti: str | None = None  # CSV; None = dati dimostrativi
    episodi: list[tuple[str, str, str]] = field(default_factory=lambda: list(dati.EPISODI))
    confronto_regimi: bool = True
    n_bootstrap: int = 2000


@dataclass
class Analisi:
    config: Configurazione
    rendimenti: pd.DataFrame
    veri: pd.Series | None
    stima: StimaRegimi  # su tutti i dati: descrizione dei regimi e proposta di oggi
    previsioni: Previsioni
    ingenuo: Previsioni
    risultati: dict[str, Risultato]
    tabella: pd.DataFrame
    episodi: pd.DataFrame
    per_regime: pd.DataFrame
    bootstrap: pd.DataFrame
    robustezza: pd.DataFrame
    confronto_regimi: pd.DataFrame | None
    selezione: pd.DataFrame
    riconoscimento: dict | None
    proposta: pd.DataFrame
    probabilita_oggi: pd.Series
    probabilita_mese: pd.Series

    @property
    def liquidita(self) -> pd.Series:
        r = self.risultati[BENCH].rendimenti
        return dati.mensili(self.rendimenti)[self.config.parametri.liquidita].loc[r.index]

    @property
    def periodi_anno(self) -> int:
        return self.stima.periodi_anno


def _episodi(risultati: dict[str, Risultato], episodi) -> pd.DataFrame:
    righe = {}
    for nome, a, b in episodi:
        riga = {}
        for s, r in risultati.items():
            x = r.rendimenti[(r.rendimenti.index.to_period("M") >= pd.Period(a, "M"))
                             & (r.rendimenti.index.to_period("M") <= pd.Period(b, "M"))]
            riga[s] = float((1 + x).prod() - 1) if len(x) else np.nan
        righe[f"{nome}"] = riga
    df = pd.DataFrame(righe).T
    return df.dropna(how="all")


def _per_regime(risultati: dict[str, Risultato], regime_mese: pd.Series) -> pd.DataFrame:
    out = {}
    for s, r in risultati.items():
        g = r.rendimenti.groupby(regime_mese.reindex(r.rendimenti.index).values)
        out[s] = (1 + g.mean()) ** 12 - 1
    df = pd.DataFrame(out)
    df["Mesi"] = risultati[BENCH].rendimenti.groupby(regime_mese.reindex(risultati[BENCH].rendimenti.index).values
                                                      ).size()
    return df


def _riconoscimento(etichette: pd.Series, veri: pd.Series) -> dict:
    """Confronto tra i regimi riconosciuti in tempo reale e quelli veri della simulazione."""
    v = veri.reindex(etichette.index)
    conf = pd.crosstab(v.rename("Vero"), etichette.rename("Riconosciuto"))
    accuratezza = float((v.values == etichette.values).mean())
    episodi = []
    blocchi = (v != v.shift()).cumsum()
    for _, g in v.groupby(blocchi):
        nome = g.iloc[0]
        if nome == "Espansione" or len(g) < 4:
            continue
        e = etichette.loc[g.index]
        presi = np.flatnonzero(e.values == nome)
        episodi.append({"Regime": nome, "Inizio": g.index[0], "Fine": g.index[-1], "Settimane": len(g),
                        "Ritardo": int(presi[0]) if len(presi) else None,
                        "Quota riconosciuta": float((e.values == nome).mean())})
    falsi = {n: int(((etichette.values == n) & (v.values == "Espansione")).sum()) for n in conf.columns
             if n != "Espansione"}
    return {"confusione": conf, "accuratezza": accuratezza, "episodi": pd.DataFrame(episodi), "falsi_allarmi": falsi,
            "settimane": len(v)}


def analizza(config: Configurazione, rendimenti: pd.DataFrame | None = None,
             veri: pd.Series | None = None) -> Analisi:
    par = config.parametri
    if rendimenti is None:
        if config.rendimenti:
            rendimenti, veri = dati.carica_csv(config.rendimenti)
        else:
            rendimenti, veri = dati.rendimenti_dimostrativi(), dati.regimi_veri()
    b = config.benchmark.reindex(rendimenti.columns).fillna(0.0)
    mancanti = [a for a in config.benchmark.index if a not in rendimenti.columns]
    if mancanti:
        raise ValueError(f"Attività del benchmark non presenti nei dati: {', '.join(mancanti)}")
    if abs(b.sum() - 1) > 1e-6:
        raise ValueError(f"I pesi del benchmark devono sommare a 1 (ora {b.sum():.4f})")
    for a in par.rischiosi + par.rifugi:
        if a not in rendimenti.columns:
            raise ValueError(f"Attività sconosciuta nella regola risk-off: {a}")
    for a in par.bande:
        if a not in rendimenti.columns:
            raise ValueError(f"Attività sconosciuta nelle bande: {a}")

    prev = walk_forward(rendimenti, par)
    ingenuo = senno_di_poi(rendimenti, par)
    risultati = {
        BENCH: simula(BENCH, "benchmark", prev, rendimenti, b, par),
        REGIMI: simula(REGIMI, "regimi", prev, rendimenti, b, par),
        MV: simula(MV, "media_varianza", prev, rendimenti, b, par),
        RISKOFF: simula(RISKOFF, "risk_off", prev, rendimenti, b, par),
        INGENUO: simula(INGENUO, "regimi", ingenuo, rendimenti, b, par),
    }
    rb = risultati[BENCH].rendimenti
    rf = dati.mensili(rendimenti)[par.liquidita].loc[rb.index]
    tab = pd.DataFrame({s: metriche(r.rendimenti, rf, rb) for s, r in risultati.items()}).T
    tab["Rotazione annua"] = [r.rotazione.mean() * 12 for r in risultati.values()]
    tab["Costi annui"] = [r.costi.mean() * 12 for r in risultati.values()]

    # regime di ogni mese: vero se noto, altrimenti quello riconosciuto più spesso nel mese
    if veri is not None:
        mese = veri.groupby(veri.index.to_period("M")).agg(lambda x: x.value_counts().index[0])
    else:
        e = prev.etichette
        mese = e.groupby(e.index.to_period("M")).agg(lambda x: x.value_counts().index[0])
    mese.index = mese.index.to_timestamp(how="end").normalize()

    boot = bootstrap_differenza(risultati[REGIMI].rendimenti, rb, rf, n=config.n_bootstrap, seed=par.seed)

    rob = []
    for banda in BANDE:
        for costo in COSTI:
            p2 = par.con_bande_scalate(banda)
            p2.costo = costo
            r2 = simula("", "regimi", prev, rendimenti, b, p2).rendimenti
            b2 = simula("", "benchmark", prev, rendimenti, b, p2).rendimenti
            m = metriche(r2, rf, b2)
            rob.append({"Banda": banda, "Costo": costo, "Extra-rendimento": m["Extra-rendimento"],
                        "Differenza di Sharpe": m["Sharpe"] - metriche(b2, rf)["Sharpe"],
                        "Massimo drawdown": m["Massimo drawdown"]})

    confronto = None
    if config.confronto_regimi:
        righe = []
        for k in (2, 3, 4):
            if k == par.n_regimi:
                pk, rk = prev, risultati[REGIMI]
            else:
                p2 = Parametri(**{**par.__dict__, "n_regimi": k})
                pk = walk_forward(rendimenti, p2)
                rk = simula("", "regimi", pk, rendimenti, b, p2)
            m = metriche(rk.rendimenti, rf, rb)
            riga = {"Regimi": k, **{x: m[x] for x in ("Rendimento annuo", "Volatilità", "Sharpe", "Massimo drawdown",
                                                         "Extra-rendimento")},
                    "Rotazione annua": rk.rotazione.mean() * 12}
            if veri is not None and k == par.n_regimi:
                riga["Accuratezza"] = float((veri.reindex(pk.etichette.index).values == pk.etichette.values).mean())
            righe.append(riga)
        confronto = pd.DataFrame(righe).set_index("Regimi")

    stima = stima_regimi(rendimenti, par.caratteristiche, par.liquidita, par.n_regimi, par.persistenza,
                         max(par.n_avvii, 6), par.seed)
    X = stima._X(rendimenti)
    sel = pd.DataFrame(selezione_regimi(X, 5, n_avvii=4, persistenza=par.persistenza, seed=par.seed)).set_index(
        "Regimi")

    oggi = stima.filtrate(rendimenti).iloc[-1]
    mese_p = pd.Series(stima.prossimo_mese(oggi.values), index=stima.nomi)
    pi = rendimenti_equilibrio(stima.cov_totale, b.values, par.avversione)
    mu, S = momenti_misti(mese_p.values, ancora(stima.medie, stima.media_totale, pi, par.restringimento),
                          stima.covarianze)
    bande = par.bande_per(b.index)
    w = ottimizza(mu, S, b.values, b.values, par.avversione, bande.values, par.costo, par.orizzonte_costi)
    w_ro = regola_risk_off(b, float(mese_p.iloc[-1]), par.soglia_crisi, bande, par.rischiosi, par.rifugi)
    proposta = pd.DataFrame({"Benchmark": b, "Per regimi": w, "Risk-off": w_ro.values}, index=b.index)
    proposta["Differenza"] = proposta["Per regimi"] - proposta["Benchmark"]
    proposta["Importo"] = proposta["Per regimi"] * config.capitale
    proposta["Operazione"] = proposta["Differenza"] * config.capitale

    return Analisi(config, rendimenti, veri, stima, prev, ingenuo, risultati, tab,
                   _episodi(risultati, config.episodi), _per_regime(risultati, mese), boot, pd.DataFrame(rob),
                   confronto, sel, _riconoscimento(prev.etichette, veri) if veri is not None else None, proposta,
                   oggi, mese_p)
