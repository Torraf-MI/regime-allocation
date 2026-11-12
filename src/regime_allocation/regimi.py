"""Riconoscimento dei regimi di mercato: stima dell'HMM sui rendimenti, nomi dei regimi, momenti per regime.

L'HMM si stima sui rendimenti in eccesso sulla liquidità di poche variabili (di default azioni, titoli di
Stato e obbligazioni societarie: rischio azionario, rischio tasso e rischio di credito), standardizzate con
media e deviazione standard della sola finestra di stima. I regimi vengono ordinati per volatilità azionaria
crescente e ricevono un nome economico: il più calmo è «Espansione», il più volatile «Crisi», quelli intermedi
«Tassi in rialzo» se in quel regime i titoli di Stato perdono, altrimenti «Rallentamento».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .hmm import ModelloHMM, stima_hmm


def eccessi(rendimenti: pd.DataFrame, liquidita: str) -> pd.DataFrame:
    """Rendimenti in eccesso sulla liquidità (la liquidità ha eccesso nullo)."""
    return rendimenti.sub(rendimenti[liquidita], axis=0)


def momenti_pesati(R: np.ndarray, pesi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media e covarianza di R con pesi non negativi per osservazione."""
    w = pesi / pesi.sum()
    m = w @ R
    D = R - m
    return m, (w[:, None] * D).T @ D


def nomi_regimi(medie_attivi: np.ndarray, vol_azioni: np.ndarray, i_obbl: int | None) -> list[str]:
    """Nomi economici per regimi già ordinati per volatilità azionaria crescente."""
    K = len(vol_azioni)
    if K == 1:
        return ["Unico regime"]
    nomi = ["Espansione"] + [""] * (K - 2) + ["Crisi"]
    intermedi = list(range(1, K - 1))
    usati: dict[str, int] = {}
    for k in intermedi:
        n = "Tassi in rialzo" if i_obbl is not None and medie_attivi[k, i_obbl] < 0 else "Rallentamento"
        usati[n] = usati.get(n, 0) + 1
        nomi[k] = n
    for n, c in usati.items():  # nomi ripetuti: si numerano
        if c > 1:
            j = 0
            for k in intermedi:
                if nomi[k] == n:
                    j += 1
                    nomi[k] = f"{n} {j}"
    return nomi


@dataclass
class StimaRegimi:
    """HMM stimato su una finestra di dati, con nomi dei regimi e momenti annui degli attivi per regime."""

    modello: ModelloHMM
    nomi: list[str]
    caratteristiche: list[str]
    centro: np.ndarray
    scala: np.ndarray
    attivi: list[str]
    medie: np.ndarray  # (K, n) rendimenti in eccesso annui per regime
    covarianze: np.ndarray  # (K, n, n) annue
    media_totale: np.ndarray
    cov_totale: np.ndarray
    periodi_anno: int
    fine: pd.Timestamp
    lisciate: pd.DataFrame  # probabilità lisciate nella finestra di stima
    liquidita: str

    @property
    def n_regimi(self) -> int:
        return self.modello.n_regimi

    def _X(self, rendimenti: pd.DataFrame) -> np.ndarray:
        return (eccessi(rendimenti, self.liquidita)[self.caratteristiche].values - self.centro) / self.scala

    def filtrate(self, rendimenti: pd.DataFrame) -> pd.DataFrame:
        """Probabilità filtrate (solo passato) su tutti i periodi passati, con i parametri di questa stima."""
        return pd.DataFrame(self.modello.filtra(self._X(rendimenti)), index=rendimenti.index, columns=self.nomi)

    def lisciate_su(self, rendimenti: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.modello.liscia(self._X(rendimenti)), index=rendimenti.index, columns=self.nomi)

    def viterbi(self, rendimenti: pd.DataFrame) -> pd.Series:
        s = self.modello.viterbi(self._X(rendimenti))
        return pd.Series(np.array(self.nomi)[s], index=rendimenti.index, name="Regime")

    @property
    def passi_mese(self) -> int:
        return max(1, round(self.periodi_anno / 12))

    def prossimo_mese(self, probabilita: np.ndarray) -> np.ndarray:
        """Quota attesa del mese successivo in ciascun regime: media delle previsioni a 1…h periodi."""
        p, A, out = np.asarray(probabilita, float), self.modello.transizioni, []
        for _ in range(self.passi_mese):
            p = p @ A
            out.append(p)
        return np.mean(out, axis=0)

    def transizioni_mensili(self) -> pd.DataFrame:
        A = np.linalg.matrix_power(self.modello.transizioni, self.passi_mese)
        return pd.DataFrame(A, index=self.nomi, columns=self.nomi)

    def durate_mesi(self) -> pd.Series:
        return pd.Series(self.modello.durate_attese() * 12 / self.periodi_anno, index=self.nomi)

    def tabella(self) -> pd.DataFrame:
        """Rendimento in eccesso e volatilità annui di ogni attività in ogni regime."""
        righe = []
        for k, n in enumerate(self.nomi):
            for j, a in enumerate(self.attivi):
                righe.append({"Regime": n, "Attività": a, "Rendimento": self.medie[k, j],
                              "Volatilità": np.sqrt(self.covarianze[k, j, j])})
        return pd.DataFrame(righe)

    def correlazione(self, a: str, b: str) -> pd.Series:
        i, j = self.attivi.index(a), self.attivi.index(b)
        c = self.covarianze
        return pd.Series(c[:, i, j] / np.sqrt(np.maximum(c[:, i, i] * c[:, j, j], 1e-18)), index=self.nomi)


def stima_regimi(rendimenti: pd.DataFrame, caratteristiche: list[str], liquidita: str, n_regimi: int = 3,
                 persistenza: float = 0.0, n_avvii: int = 4, seed: int = 0,
                 precedente: StimaRegimi | None = None) -> StimaRegimi:
    """Stima l'HMM su tutti i dati passati in ``rendimenti`` (nessun dato successivo viene usato)."""
    from .dati import periodi_per_anno

    mancanti = [c for c in caratteristiche + [liquidita] if c not in rendimenti.columns]
    if mancanti:
        raise ValueError(f"Attività non presenti nei dati: {', '.join(mancanti)}")
    ppa = periodi_per_anno(rendimenti.index)
    E = eccessi(rendimenti, liquidita)
    Xr = E[caratteristiche].values
    centro, scala = Xr.mean(axis=0), Xr.std(axis=0)
    scala = np.where(scala > 0, scala, 1.0)
    X = (Xr - centro) / scala
    iniziale = None
    if precedente is not None and precedente.n_regimi == n_regimi:
        m = precedente.modello  # ripartenza dalla stima precedente, riportata alla nuova standardizzazione
        f = precedente.scala / scala
        medie = ((m.medie * precedente.scala + precedente.centro) - centro) / scala
        iniziale = ModelloHMM(m.iniziale.copy(), m.transizioni.copy(), medie, m.covarianze * np.outer(f, f))
    mod = stima_hmm(X, n_regimi, n_avvii=n_avvii, persistenza=persistenza, seed=seed, iniziale=iniziale)
    mod = mod.riordina(np.argsort(mod.covarianze[:, 0, 0]))
    gamma = mod.liscia(X)
    R = E.values
    medie = np.empty((n_regimi, R.shape[1]))
    cov = np.empty((n_regimi, R.shape[1], R.shape[1]))
    for k in range(n_regimi):
        medie[k], cov[k] = momenti_pesati(R, gamma[:, k] + 1e-12)
    mt, ct = momenti_pesati(R, np.ones(len(R)))
    attivi = list(rendimenti.columns)
    i_obbl = attivi.index(caratteristiche[1]) if len(caratteristiche) > 1 else None
    nomi = nomi_regimi(medie, np.sqrt(cov[:, attivi.index(caratteristiche[0]), attivi.index(caratteristiche[0])]),
                       i_obbl)
    return StimaRegimi(mod, nomi, list(caratteristiche), centro, scala, attivi, medie * ppa, cov * ppa, mt * ppa,
                       ct * ppa, ppa, rendimenti.index[-1], pd.DataFrame(gamma, index=rendimenti.index, columns=nomi),
                       liquidita)
