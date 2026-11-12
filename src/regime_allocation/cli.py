"""Riga di comando: regimi di mercato, backtest fuori campione, proposta di allocazione e report.

Esempi::

    regime-allocation --demo
    regime-allocation --config esempi/cliente.json --output output/report.html
    regime-allocation --demo --veloce --grafici docs/img
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import configurazione
from .analisi import REGIMI, STRATEGIE, analizza
from .report import _eur, _pct, build_report, salva_grafici

NOTA_DEMO = ("Analisi svolta su dati simulati: i rendimenti settimanali ricalcano per date e ordine di grandezza "
             "i mercati in euro dal 2000, ma non sono quelli di alcun indice reale. I regimi veri della simulazione "
             "servono solo a misurare il modello, che non li conosce.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="regime-allocation", description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="File JSON con dati, benchmark e parametri")
    src.add_argument("--demo", action="store_true", help="Usa i dati e il benchmark dimostrativi")
    p.add_argument("--cliente", help="Nome del cliente nel report")
    p.add_argument("--capitale", type=float, help="Capitale del cliente in euro")
    p.add_argument("--output", default="output/report.html", help="Percorso del report HTML")
    p.add_argument("--grafici", help="Cartella dove salvare anche i grafici PNG (richiede kaleido)")
    p.add_argument("--offline", action="store_true", help="Include plotly.js nel report (funziona senza rete)")
    p.add_argument("--veloce", action="store_true", help="Salta il confronto con 2 e 4 regimi")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = configurazione.esempio() if args.demo else configurazione.carica(args.config)
    if args.cliente:
        cfg.cliente = args.cliente
    if args.capitale:
        cfg.capitale = args.capitale
    if args.veloce:
        cfg.confronto_regimi = False
    a = analizza(cfg)
    nota = NOTA_DEMO if cfg.rendimenti is None else ""
    path = build_report(a, args.output, data_note=nota, offline=args.offline)
    if args.grafici:
        salva_grafici(a, args.grafici)

    pd.set_option("display.width", 200)
    print(f"Regimi stimati: {', '.join(a.stima.nomi)}")
    print("Oggi: " + ", ".join(f"{n} {_pct(x, 0)}" for n, x in a.probabilita_oggi.items()))
    if a.riconoscimento is not None:
        print(f"Regimi riconosciuti correttamente fuori campione: {_pct(a.riconoscimento['accuratezza'], 0)}")
    t = a.tabella.loc[STRATEGIE, ["Rendimento annuo", "Volatilità", "Sharpe", "Massimo drawdown", "Extra-rendimento",
                                  "Rotazione annua"]]
    print("\nBacktest fuori campione")
    print(t.round(3).to_string())
    quota = float((a.bootstrap["Extra-rendimento"] > 0).mean())
    print(f"\nExtra-rendimento positivo nel {_pct(quota, 0)} dei campioni bootstrap")
    print(f"\nProposta per il prossimo mese ({REGIMI})")
    for i, r in a.proposta.iterrows():
        print(f"  {i:<26} {_pct(r['Benchmark'], 1):>6} → {_pct(r['Per regimi'], 1):>6}   {_eur(r['Operazione']):>12}")
    print(f"\nReport salvato in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
