"""Uso della libreria in poche righe: regimi stimati, probabilità di oggi, backtest fuori campione."""

from regime_allocation import Parametri, dati, metriche, simula, stima_regimi, walk_forward
from regime_allocation.configurazione import CARATTERISTICHE, RIFUGI, RISCHIOSI

rendimenti = dati.rendimenti_dimostrativi()  # settimanali, sei classi di attività dal 2000

# 1. Un HMM a tre regimi su azioni, titoli di Stato e obbligazioni societarie
s = stima_regimi(rendimenti, CARATTERISTICHE, dati.LIQUIDITA, n_regimi=3)
print("Regimi:", s.nomi)
print("Durata media (mesi):", s.durate_mesi().round(1).to_dict())
print("Correlazione azioni-Stato:", s.correlazione("Azioni globali", "Titoli di Stato euro").round(2).to_dict())
oggi = s.filtrate(rendimenti).iloc[-1]
print("Probabilità oggi:", oggi.round(3).to_dict())
print("Quota attesa del prossimo mese:", dict(zip(s.nomi, s.prossimo_mese(oggi.values).round(3).tolist())))

# 2. Backtest walk-forward: il modello vede solo il passato a ogni fine mese
par = Parametri(list(CARATTERISTICHE), dati.LIQUIDITA, rischiosi=list(RISCHIOSI), rifugi=list(RIFUGI),
                inizio_test="2007-01", banda=0.10)
prev = walk_forward(rendimenti, par)
bench = simula("Benchmark", "benchmark", prev, rendimenti, dati.BENCHMARK, par)
regimi = simula("Per regimi", "regimi", prev, rendimenti, dati.BENCHMARK, par)
rf = dati.mensili(rendimenti)[dati.LIQUIDITA].loc[bench.rendimenti.index]
for r in (bench, regimi):
    m = metriche(r.rendimenti, rf, bench.rendimenti)
    print(f"{r.nome:<12} rendimento {m['Rendimento annuo']:.2%}, Sharpe {m['Sharpe']:.2f}, "
          f"drawdown massimo {m['Massimo drawdown']:.1%}")
