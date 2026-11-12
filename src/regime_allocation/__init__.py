"""Regimi di mercato con Hidden Markov Model e allocazione tattica validata fuori campione."""

from .allocazione import ancora, momenti_misti, ottimizza, regola_risk_off, rendimenti_equilibrio
from .analisi import Analisi, Configurazione, analizza
from .backtest import Parametri, senno_di_poi, simula, walk_forward
from .hmm import ModelloHMM, selezione_regimi, stima_hmm
from .metriche import bootstrap_differenza, metriche
from .regimi import StimaRegimi, stima_regimi

__all__ = [
    "Analisi", "Configurazione", "ModelloHMM", "Parametri", "StimaRegimi", "analizza", "ancora", "bootstrap_differenza",
    "metriche", "momenti_misti", "ottimizza", "regola_risk_off", "rendimenti_equilibrio", "selezione_regimi",
    "senno_di_poi", "simula", "stima_hmm", "stima_regimi", "walk_forward",
]
__version__ = "0.1.0"
