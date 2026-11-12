"""Backtest walk-forward: a ogni fine mese si usano solo i dati fino a quel giorno.

Ogni ``rifai_ogni`` mesi l'HMM viene ristimato su tutta la storia disponibile; nei mesi intermedi i parametri
restano fermi e si aggiornano solo le probabilità filtrate con le nuove osservazioni. Dalle probabilità di
oggi si prevede la quota del mese prossimo in ciascun regime, si costruisce il portafoglio e lo si tiene fermo
per il mese, pagando i costi sulla differenza tra i nuovi pesi e quelli derivati con i rendimenti del mese
precedente.

Il confronto con il **backtest ingenuo** mostra l'errore più comune: stimare il modello una volta sola su tutto
il campione e usare le probabilità lisciate del mese stesso in cui si investe, che conoscono già il futuro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .allocazione import ancora, momenti_misti, ottimizza, regola_risk_off, rendimenti_equilibrio
from .dati import mensili, periodi_per_anno
from .regimi import StimaRegimi, stima_regimi


@dataclass
class Parametri:
    caratteristiche: list[str]
    liquidita: str
    n_regimi: int = 3
    persistenza: float = 0.0
    inizio_test: str = "2007-01"
    rifai_ogni: int = 12
    n_avvii: int = 4
    banda: float = 0.10
    bande: dict[str, float] = field(default_factory=dict)  # bande diverse per singole attività
    avversione: float = 5.0
    costo: float = 0.0010
    orizzonte_costi: float = 3.0
    restringimento: float = 0.5
    soglia_crisi: float = 0.5
    rischiosi: list[str] = field(default_factory=list)
    rifugi: list[str] = field(default_factory=list)
    seed: int = 0

    def bande_per(self, attivi) -> pd.Series:
        """Banda di ogni attività: quella generale, salvo eccezioni in ``bande``."""
        return pd.Series({a: float(self.bande.get(a, self.banda)) for a in attivi})

    def con_bande_scalate(self, banda: float) -> Parametri:
        """Copia con banda generale ``banda`` e le eccezioni riscalate nella stessa proporzione."""
        f = banda / self.banda if self.banda > 0 else 0.0
        return Parametri(**{**self.__dict__, "banda": banda, "bande": {a: x * f for a, x in self.bande.items()}})


@dataclass
class Previsioni:
    """Informazioni disponibili a ogni data di decisione (fine mese)."""

    date: pd.DatetimeIndex
    prossimo: np.ndarray  # (D, K) quota attesa del mese successivo in ciascun regime
    filtrate: np.ndarray  # (D, K) probabilità filtrate alla data
    nomi: list[list[str]]
    medie: list[np.ndarray]
    covarianze: list[np.ndarray]
    media_totale: list[np.ndarray]
    cov_totale: list[np.ndarray]
    settimanali: pd.DataFrame  # probabilità filtrate fuori campione di ogni periodo, colonne 0..K-1
    etichette: pd.Series  # nome del regime più probabile, fuori campione
    stime: list[StimaRegimi]

    @property
    def p_crisi(self) -> pd.Series:
        return pd.Series(self.prossimo[:, -1], index=self.date)


@dataclass
class Risultato:
    nome: str
    rendimenti: pd.Series  # mensili, al netto dei costi
    pesi: pd.DataFrame  # pesi scelti a ogni data di decisione
    rotazione: pd.Series  # somma dei valori assoluti delle operazioni
    costi: pd.Series


def date_decisione(rendimenti: pd.DataFrame, inizio_test: str) -> pd.DatetimeIndex:
    """Fine dei mesi dal mese prima dell'inizio del test al penultimo mese dei dati."""
    mesi = mensili(rendimenti).index
    primo = pd.Period(inizio_test, "M") - 1
    d = mesi[mesi.to_period("M") >= primo][:-1]
    if len(d) < 12:
        raise ValueError("Periodo di test troppo corto: servono almeno 12 mesi dopo l'inizio del test")
    if (rendimenti.index <= d[0]).sum() < 3 * periodi_per_anno(rendimenti.index):
        raise ValueError("Servono almeno tre anni di dati prima dell'inizio del test per stimare il modello")
    return d


def walk_forward(rendimenti: pd.DataFrame, par: Parametri) -> Previsioni:
    """Probabilità dei regimi e momenti per regime calcolati in tempo reale, senza dati futuri."""
    date = date_decisione(rendimenti, par.inizio_test)
    stima: StimaRegimi | None = None
    out = {k: [] for k in ("prossimo", "filtrate", "nomi", "medie", "cov", "mt", "ct")}
    stime, blocchi = [], []
    ultima = None
    for i, d in enumerate(date):
        passato = rendimenti.loc[:d]
        if stima is not None:
            # i periodi del mese appena trascorso si classificano con il modello stimato prima di vederli
            f = stima.filtrate(passato)
            nuovi = f.loc[f.index > ultima]
            blocchi.append((nuovi.values, nuovi.index, stima.nomi))
        if stima is None or i % par.rifai_ogni == 0:
            stima = stima_regimi(passato, par.caratteristiche, par.liquidita, par.n_regimi, par.persistenza,
                                 par.n_avvii, par.seed, precedente=stima)
            stime.append(stima)
        f = stima.filtrate(passato)
        ultima = d
        p = f.values[-1]
        out["prossimo"].append(stima.prossimo_mese(p))
        out["filtrate"].append(p)
        out["nomi"].append(stima.nomi)
        out["medie"].append(stima.medie)
        out["cov"].append(stima.covarianze)
        out["mt"].append(stima.media_totale)
        out["ct"].append(stima.cov_totale)
    # probabilità filtrate di ogni periodo con i parametri disponibili in quel momento
    if stima is not None:
        resto = rendimenti.loc[rendimenti.index > ultima]
        if len(resto):
            f = stima.filtrate(rendimenti).loc[resto.index]
            blocchi.append((f.values, f.index, stima.nomi))
    sett = pd.DataFrame(np.vstack([b[0] for b in blocchi]), index=np.concatenate([b[1] for b in blocchi]))
    sett.index = pd.DatetimeIndex(sett.index)
    etichette = pd.Series(np.concatenate([np.array(b[2])[b[0].argmax(axis=1)] for b in blocchi]), index=sett.index,
                          name="Regime")
    return Previsioni(date, np.array(out["prossimo"]), np.array(out["filtrate"]), out["nomi"], out["medie"],
                      out["cov"], out["mt"], out["ct"], sett, etichette, stime)


def senno_di_poi(rendimenti: pd.DataFrame, par: Parametri) -> Previsioni:
    """Backtest ingenuo, come se ne vedono spesso: un'unica stima su tutto il campione e, per ogni mese,
    le probabilità lisciate dello stesso mese in cui si investe. Il modello conosce così sia il futuro dei
    parametri sia il regime del mese che deve prevedere."""
    date = date_decisione(rendimenti, par.inizio_test)
    s = stima_regimi(rendimenti, par.caratteristiche, par.liquidita, par.n_regimi, par.persistenza,
                     max(par.n_avvii, 6), par.seed)
    g = s.lisciate.reindex(date, method="ffill").values
    mese = s.lisciate.groupby(s.lisciate.index.to_period("M")).mean()
    successivo = mese.reindex(date.to_period("M") + 1).values
    D = len(date)
    return Previsioni(date, successivo, g, [s.nomi] * D, [s.medie] * D,
                      [s.covarianze] * D, [s.media_totale] * D, [s.cov_totale] * D, s.lisciate.set_axis(
                          range(s.n_regimi), axis=1), s.lisciate.idxmax(axis=1).rename("Regime"), [s])


def pesi_obiettivo(metodo: str, prev: Previsioni | None, i: int, benchmark: pd.Series, precedente: np.ndarray,
                   par: Parametri) -> np.ndarray:
    b = benchmark.values
    if metodo == "benchmark":
        return b.copy()
    if metodo == "risk_off":
        return regola_risk_off(benchmark, float(prev.prossimo[i, -1]), par.soglia_crisi,
                               par.bande_per(benchmark.index), par.rischiosi, par.rifugi).values
    if metodo == "regimi":
        pi = rendimenti_equilibrio(prev.cov_totale[i], b, par.avversione)
        medie = ancora(prev.medie[i], prev.media_totale[i], pi, par.restringimento)
        mu, S = momenti_misti(prev.prossimo[i], medie, prev.covarianze[i])
    elif metodo == "media_varianza":
        mu, S = prev.media_totale[i], prev.cov_totale[i]
    else:
        raise ValueError(f"Metodo sconosciuto: {metodo}")
    return ottimizza(mu, S, b, precedente, par.avversione, par.bande_per(benchmark.index).values, par.costo,
                     par.orizzonte_costi)


def simula(nome: str, metodo: str, prev: Previsioni | None, rendimenti: pd.DataFrame, benchmark: pd.Series,
           par: Parametri, date: pd.DatetimeIndex | None = None) -> Risultato:
    """Applica una regola di allocazione alle date di decisione e misura i rendimenti netti del mese dopo."""
    R = mensili(rendimenti)[benchmark.index]
    date = prev.date if prev is not None else (date if date is not None else date_decisione(rendimenti,
                                                                                          par.inizio_test))
    pos = R.index.get_indexer(date)
    attuali = benchmark.values.copy()
    r_out, pesi, rot, cst = [], [], [], []
    for i in range(len(date)):
        w = pesi_obiettivo(metodo, prev, i, benchmark, attuali, par)
        t = float(np.abs(w - attuali).sum()) if i else 0.0
        r = R.values[pos[i] + 1]
        lordo = float(w @ r)
        c = par.costo * t
        r_out.append(lordo - c)
        pesi.append(w)
        rot.append(t)
        cst.append(c)
        attuali = w * (1 + r) / (1 + lordo)
    mesi = R.index[pos + 1]
    return Risultato(nome, pd.Series(r_out, index=mesi, name=nome),
                     pd.DataFrame(pesi, index=date, columns=benchmark.index), pd.Series(rot, index=date),
                     pd.Series(cst, index=mesi))
