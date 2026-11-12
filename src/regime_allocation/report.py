"""Grafici interattivi (plotly) e report HTML per il cliente."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.subplots import make_subplots

from . import dati
from .analisi import BENCH, INGENUO, MV, REGIMI, RISKOFF, STRATEGIE, Analisi
from .metriche import crescita, drawdown

BLU, ARANCIO, ACQUA, VIOLA = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
BLU_SCURO, ROSSO, GIALLO, GRIGIO = "#1c5cab", "#e34948", "#e8a317", "#8c8b86"
TESTO, TESTO_SECONDARIO, GRIGLIA = "#222222", "#52514e", "#e6e5e1"
COLORE_STRATEGIA = {BENCH: GRIGIO, REGIMI: BLU_SCURO, MV: VIOLA, RISKOFF: ACQUA, INGENUO: ARANCIO}
COLORI_ATTIVI = ["#1c5cab", "#7fb0ea", "#8a5cd1", "#1baf7a", "#e8a317", "#b5b3ad", "#eb6834", "#c9a227"]
PLOTLY_CDN = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"


def colore_regime(nome: str) -> str:
    if nome.startswith("Espansione") or nome == "Unico regime":
        return ACQUA
    if nome.startswith("Crisi"):
        return ROSSO
    if nome.startswith("Tassi"):
        return GIALLO
    return VIOLA


def _rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{a})"


def _eur(x: float, dec: int = 0) -> str:
    s = f"{abs(x):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{'−' if x < -0.5 * 10**-dec else ''}{s} €"


def _pct(x: float, dec: int = 1, segno: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x * 100:+.{dec}f}%" if segno else f"{x * 100:.{dec}f}%"
    return s.replace(".", ",").replace("-", "−")


def _num(x: float, dec: int = 2, segno: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:+,.{dec}f}" if segno else f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def _layout(fig: go.Figure, titolo: str, altezza: int = 430) -> go.Figure:
    fig.update_layout(
        title={"text": titolo, "x": 0, "font": {"size": 15, "color": TESTO}},
        height=altezza, template="plotly_white",
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12, "color": TESTO_SECONDARIO},
        margin={"l": 70, "r": 20, "t": 60, "b": 50}, separators=",.",
        hoverlabel={"bgcolor": "white", "font_color": TESTO}, legend={"orientation": "h", "y": -0.16, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRIGLIA, zeroline=False)
    fig.update_yaxes(gridcolor=GRIGLIA, zerolinecolor="#bdbcb6")
    return fig


def _blocchi(etichette: pd.Series):
    """Intervalli consecutivi con la stessa etichetta: (nome, inizio, fine)."""
    g = (etichette != etichette.shift()).cumsum()
    for _, x in etichette.groupby(g):
        yield x.iloc[0], x.index[0], x.index[-1]


def _nomi_prevalenti(a: Analisi) -> list[str]:
    tuple_ = pd.Series([tuple(n) for n in a.previsioni.nomi]).value_counts()
    return list(tuple_.index[0])


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
def grafico_regimi(a: Analisi) -> go.Figure:
    """Azioni con lo sfondo colorato dal regime riconosciuto in tempo reale, e i regimi veri sotto."""
    p = a.config.parametri
    az = p.caratteristiche[0]
    e = a.previsioni.etichette
    c = (1 + a.rendimenti[az].loc[e.index[0]:]).cumprod() * 100
    righe = 2 if a.veri is not None else 1
    fig = make_subplots(rows=righe, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.88, 0.12] if righe == 2 else [1.0],
                        specs=[[{"secondary_y": True}]] + ([[{}]] if righe == 2 else []))
    largo = (7 if a.periodi_anno == 52 else 31) * 86_400_000
    for nome in dict.fromkeys(e.values):
        m = e == nome
        fig.add_bar(x=e.index[m], y=np.ones(m.sum()), width=largo, name=f"Riconosciuto: {nome}",
                    marker={"color": _rgba(colore_regime(nome), 0.30), "line": {"width": 0}}, hoverinfo="skip",
                    row=1, col=1, secondary_y=False)
    fig.add_scatter(x=c.index, y=c, name=az, line={"color": TESTO, "width": 1.6}, row=1, col=1, secondary_y=True,
                    hovertemplate="%{x|%d/%m/%Y}: %{y:.1f}<extra></extra>")
    if a.veri is not None:
        v = a.veri.loc[e.index[0]:]
        for nome in dict.fromkeys(v.values):
            m = v == nome
            fig.add_bar(x=v.index[m], y=np.ones(m.sum()), width=largo, name=f"Vero: {nome}", showlegend=False,
                        marker={"color": colore_regime(nome), "line": {"width": 0}}, row=2, col=1,
                        hovertemplate="%{x|%d/%m/%Y}<extra>vero: " + nome + "</extra>")
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, 1], title_text="veri", row=2, col=1)
    fig.update_layout(barmode="overlay", bargap=0)
    _layout(fig, "I regimi di mercato riconosciuti in tempo reale, senza guardare il futuro", 520)
    # barre sull'asse principale (nascosto), linea sull'asse secondario portato a sinistra: la linea resta sopra
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, secondary_y=False, row=1, col=1)
    tick = [v for v in (25, 50, 75, 100, 150, 200, 300, 400, 600, 800) if c.min() / 1.3 <= v <= c.max() * 1.3]
    fig.update_yaxes(type="log", title_text=f"{az} (base 100)", side="left", gridcolor=GRIGLIA, secondary_y=True,
                     tickvals=tick, row=1, col=1)
    fig.update_layout(legend={"y": -0.12})
    return fig


def grafico_probabilita(a: Analisi) -> go.Figure:
    s = a.previsioni.settimanali
    nomi = _nomi_prevalenti(a)
    fig = go.Figure()
    for k in range(s.shape[1]):
        n = nomi[k] if k < len(nomi) else f"Regime {k + 1}"
        fig.add_scatter(x=s.index, y=s[k], name=n, stackgroup="p", mode="none",
                        fillcolor=_rgba(colore_regime(n), 0.75), hovertemplate="%{x|%d/%m/%Y}: %{y:.0%}<extra>" + n +
                        "</extra>")
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title="Probabilità filtrata")
    return _layout(fig, "Probabilità dei regimi ogni settimana, con le sole informazioni disponibili a quella data",
                   400)


def grafico_caratteristiche(a: Analisi) -> go.Figure:
    t = a.stima.tabella()
    t = t[t["Attività"] != a.config.parametri.liquidita]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Rendimento in eccesso annuo", "Volatilità annua"))
    for n in a.stima.nomi:
        x = t[t["Regime"] == n]
        for col, campo in ((1, "Rendimento"), (2, "Volatilità")):
            fig.add_bar(x=x["Attività"].str.replace(" ", "<br>", n=1), y=x[campo], name=n,
                        marker_color=colore_regime(n), showlegend=col == 1,
                        legendgroup=n, row=1, col=col, hovertemplate="%{x}: %{y:.1%}<extra>" + n + "</extra>")
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=0)
    fig.update_layout(barmode="group")
    _layout(fig, "Come si comporta ogni classe di attività in ciascun regime (stima su tutto il campione)", 460)
    fig.update_layout(legend={"y": -0.22})
    return fig


def grafico_crescita(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for s in STRATEGIE:
        r = a.risultati[s].rendimenti
        c = pd.concat([pd.Series([1.0], index=[r.index[0] - pd.offsets.MonthEnd(1)]), crescita(r)]) * 100
        fig.add_scatter(x=c.index, y=c, name=s, line={"color": COLORE_STRATEGIA[s], "width": 2.8 if s == REGIMI
                                                       else 1.6, "dash": "dash" if s == INGENUO else "solid"},
                        hovertemplate="%{x|%m/%Y}: %{y:.1f}<extra>" + s + "</extra>")
    fig.update_yaxes(title="Valore (base 100)")
    return _layout(fig, "Crescita di 100 investiti: backtest fuori campione al netto dei costi", 460)


def grafico_drawdown(a: Analisi) -> go.Figure:
    fig = go.Figure()
    for s in (BENCH, REGIMI, RISKOFF):
        d = drawdown(a.risultati[s].rendimenti)
        fig.add_scatter(x=d.index, y=d, name=s, line={"color": COLORE_STRATEGIA[s], "width": 1.8},
                        fill="tozeroy" if s == BENCH else None,
                        fillcolor=_rgba(GRIGIO, 0.2) if s == BENCH else None,
                        hovertemplate="%{x|%m/%Y}: %{y:.1%}<extra>" + s + "</extra>")
    fig.update_yaxes(tickformat=".0%", title="Perdita dal massimo")
    return _layout(fig, "Perdite dal massimo precedente", 380)


def grafico_pesi(a: Analisi) -> go.Figure:
    w = a.risultati[REGIMI].pesi
    fig = go.Figure()
    for i, c in enumerate(w.columns):
        fig.add_scatter(x=w.index, y=w[c], name=c, stackgroup="w", mode="none",
                        fillcolor=COLORI_ATTIVI[i % len(COLORI_ATTIVI)],
                        hovertemplate="%{x|%m/%Y}: %{y:.1%}<extra>" + c + "</extra>")
    pc = a.previsioni.p_crisi
    fig.add_scatter(x=pc.index, y=pc, name="Probabilità di crisi (mese dopo)", line={"color": ROSSO, "width": 1.4},
                    yaxis="y2", hovertemplate="%{x|%m/%Y}: %{y:.0%}<extra>crisi</extra>")
    fig.update_layout(yaxis={"tickformat": ".0%", "range": [0, 1], "title": "Pesi"},
                      yaxis2={"overlaying": "y", "side": "right", "range": [0, 1], "tickformat": ".0%",
                              "showgrid": False, "title": "Probabilità di crisi"})
    _layout(fig, "Pesi dell'allocazione per regimi a ogni fine mese", 460)
    fig.update_layout(margin={"r": 70})
    return fig


def grafico_episodi(a: Analisi) -> go.Figure:
    e = a.episodi
    fig = go.Figure()
    for s in (BENCH, REGIMI, RISKOFF, INGENUO):
        fig.add_bar(x=e.index, y=e[s], name=s, marker_color=COLORE_STRATEGIA[s],
                    marker_pattern_shape="/" if s == INGENUO else "", hovertemplate="%{x}: %{y:.1%}<extra>" + s +
                    "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", title="Rendimento nel periodo")
    return _layout(fig, "Le crisi una per una", 430)


def grafico_bootstrap(a: Analisi) -> go.Figure:
    b = a.bootstrap
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Extra-rendimento annuo sul benchmark", "Differenza di indice di Sharpe"))
    for col, c, fmt in ((1, "Extra-rendimento", ".1%"), (2, "Differenza di Sharpe", ".2f")):
        fig.add_histogram(x=b[c], nbinsx=50, marker_color=BLU, opacity=0.8, showlegend=False, row=1, col=col,
                          hovertemplate="%{x}: %{y} campioni<extra></extra>")
        fig.add_vline(x=0, line={"color": ROSSO, "width": 1.5}, row=1, col=col)
        fig.update_xaxes(tickformat=fmt, row=1, col=col)
    fig.update_yaxes(title_text="Campioni", row=1, col=1)
    return _layout(fig, f"Fortuna o capacità? {len(b):,} storie ricampionate a blocchi".replace(",", "."), 380)


def grafico_robustezza(a: Analisi) -> go.Figure:
    t = a.robustezza.pivot(index="Banda", columns="Costo", values="Extra-rendimento")
    z = t.values
    lim = float(np.nanmax(np.abs(z))) or 0.01
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c * 1e4:.0f} pb" for c in t.columns], y=[_pct(b, 0) for b in t.index],
        colorscale=[[0, ROSSO], [0.5, "#f7f7f5"], [1, BLU_SCURO]], zmin=-lim, zmax=lim,
        text=[[_pct(v, 1, True) for v in r] for r in z], texttemplate="%{text}", showscale=False,
        hovertemplate="banda %{y}, costo %{x}: %{text}<extra></extra>"))
    fig.update_xaxes(title="Costo di transazione per unità scambiata")
    fig.update_yaxes(title="Banda di scostamento dal benchmark")
    return _layout(fig, "Extra-rendimento annuo al variare della libertà tattica e dei costi", 400)


def grafico_selezione(a: Analisi) -> go.Figure:
    s = a.selezione
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s["BIC"], name="BIC", mode="lines+markers", line={"color": BLU_SCURO, "width": 2.5})
    fig.add_scatter(x=s.index, y=s["AIC"], name="AIC", mode="lines+markers", line={"color": GRIGIO, "width": 1.5,
                                                                                    "dash": "dot"})
    k = int(s["BIC"].idxmin())
    fig.add_annotation(x=k, y=float(s.loc[k, "BIC"]), text=f"minimo BIC: {k} regimi", showarrow=True, arrowhead=2,
                       ay=-40)
    fig.update_xaxes(title="Numero di regimi", dtick=1)
    fig.update_yaxes(title="Criterio (più basso è meglio)")
    return _layout(fig, "Quanti regimi? Verosimiglianza penalizzata per il numero di parametri", 360)


def grafico_proposta(a: Analisi) -> go.Figure:
    p = a.proposta
    fig = go.Figure()
    fig.add_bar(y=p.index, x=p["Benchmark"], name="Benchmark strategico", orientation="h", marker_color="#b5b3ad",
                hovertemplate="%{y}: %{x:.1%}<extra>benchmark</extra>")
    fig.add_bar(y=p.index, x=p["Per regimi"], name="Proposta per il prossimo mese", orientation="h",
                marker_color=BLU_SCURO, hovertemplate="%{y}: %{x:.1%}<extra>proposta</extra>")
    fig.update_layout(barmode="group", yaxis={"autorange": "reversed"})
    fig.update_xaxes(tickformat=".0%")
    _layout(fig, "Allocazione proposta oggi rispetto al benchmark", 400)
    fig.update_layout(margin={"l": 170})
    return fig


def _disponibili(a: Analisi) -> dict:
    return {"regimi": grafico_regimi, "probabilita": grafico_probabilita, "caratteristiche": grafico_caratteristiche,
            "selezione": grafico_selezione, "crescita": grafico_crescita, "drawdown": grafico_drawdown,
            "pesi": grafico_pesi, "episodi": grafico_episodi, "bootstrap": grafico_bootstrap,
            "robustezza": grafico_robustezza, "proposta": grafico_proposta}


def salva_grafici(a: Analisi, cartella: str | Path, scala: float = 2.0) -> list[Path]:
    """Esporta i grafici in PNG (richiede kaleido)."""
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    percorsi = []
    for nome, f in _disponibili(a).items():
        p = cartella / f"{nome}.png"
        f(a).write_image(p, width=1000, scale=scala)
        percorsi.append(p)
    return percorsi


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _segno(x: float) -> str:
    return "pos" if x > 0.0005 else "neg" if x < -0.0005 else ""


def contesto(a: Analisi) -> dict:
    """Numeri e frasi del report, già formattati."""
    c, p, t = a.config, a.config.parametri, a.tabella
    rb, rr = t.loc[BENCH], t.loc[REGIMI]
    boot = a.bootstrap
    q = boot.quantile([0.05, 0.95])
    oggi = a.probabilita_oggi
    prev = a.probabilita_mese
    dominante = str(oggi.idxmax())
    test0, test1 = a.risultati[BENCH].rendimenti.index[[0, -1]]
    colonne = ["Rendimento annuo", "Volatilità", "Sharpe", "Massimo drawdown", "CVaR 95% mensile", "Extra-rendimento",
               "Tracking error", "Information ratio", "Rotazione annua"]
    fmt = {"Sharpe": lambda x: _num(x, 2), "Information ratio": lambda x: _num(x, 2) if not np.isnan(x) else "–",
           "Extra-rendimento": lambda x: _pct(x, 2, True)}
    strategie = [{"nome": s, "valori": [fmt.get(k, lambda x: _pct(x, 1))(t.loc[s, k]) for k in colonne],
                  "evidenza": s == REGIMI, "ingenuo": s == INGENUO} for s in STRATEGIE]
    episodi = [{"nome": n, "valori": [_pct(a.episodi.loc[n, s], 1, True) for s in STRATEGIE],
                "segni": [_segno(a.episodi.loc[n, s]) for s in STRATEGIE]} for n in a.episodi.index]
    per_regime = [{"nome": n, "mesi": int(a.per_regime.loc[n, "Mesi"]),
                   "valori": [_pct(a.per_regime.loc[n, s], 1, True) for s in STRATEGIE],
                   "segni": [_segno(a.per_regime.loc[n, s]) for s in STRATEGIE]} for n in a.per_regime.index]
    tab = a.stima.tabella()
    attivi = [x for x in a.stima.attivi if x != p.liquidita]
    regimi = []
    dur = a.stima.durate_mesi()
    staz = a.stima.modello.stazionaria()
    corr = a.stima.correlazione(p.caratteristiche[0], p.caratteristiche[1]) if len(p.caratteristiche) > 1 else None
    for k, n in enumerate(a.stima.nomi):
        x = tab[tab["Regime"] == n].set_index("Attività")
        regimi.append({"nome": n, "colore": colore_regime(n), "durata": _num(dur[n], 1), "quota": _pct(staz[k], 0),
                       "rend": [_pct(x.loc[s, "Rendimento"], 1, True) for s in attivi],
                       "segni": [_segno(x.loc[s, "Rendimento"]) for s in attivi],
                       "vol": [_pct(x.loc[s, "Volatilità"], 0) for s in attivi],
                       "corr": _num(corr[n], 2) if corr is not None else "–"})
    A = a.stima.transizioni_mensili()
    transizioni = [{"da": i, "valori": [_pct(A.loc[i, j], 0) for j in A.columns]} for i in A.index]
    proposta = [{"attivo": i, "bench": _pct(r["Benchmark"], 1), "prop": _pct(r["Per regimi"], 1),
                 "diff": _pct(r["Differenza"], 1, True), "segno": _segno(r["Differenza"]),
                 "importo": _eur(r["Importo"]),
                 "op": _eur(r["Operazione"]) if abs(r["Operazione"]) >= 1 else "–",
                 "riskoff": _pct(r["Risk-off"], 1)} for i, r in a.proposta.iterrows()]
    sel = a.selezione
    rob = a.robustezza
    base = rob[(np.isclose(rob["Banda"], p.banda)) & (np.isclose(rob["Costo"], p.costo))]
    soglia_costi = None
    fila = rob[np.isclose(rob["Banda"], p.banda)].sort_values("Costo")
    neg = fila[fila["Extra-rendimento"] <= 0]
    if len(neg):
        soglia_costi = f"{neg['Costo'].iloc[0] * 1e4:.0f}"
    ric = None
    if a.riconoscimento is not None:
        r = a.riconoscimento
        ep = r["episodi"]
        ric = {"accuratezza": _pct(r["accuratezza"], 0), "settimane": r["settimane"],
               "confusione": r["confusione"], "falsi": r["falsi_allarmi"],
               "ritardo_medio": _num(ep["Ritardo"].dropna().mean(), 1) if len(ep) else "–",
               "episodi": [{"regime": e["Regime"], "colore": colore_regime(e["Regime"]),
                            "inizio": f"{e['Inizio']:%d/%m/%Y}", "fine": f"{e['Fine']:%d/%m/%Y}",
                            "settimane": int(e["Settimane"]),
                            "ritardo": "mai" if pd.isna(e["Ritardo"]) else f"{int(e['Ritardo'])}",
                            "quota": _pct(e["Quota riconosciuta"], 0)} for _, e in ep.iterrows()]}
    conf_k = None
    if a.confronto_regimi is not None:
        conf_k = [{"k": k, "rend": _pct(r["Rendimento annuo"], 1), "vol": _pct(r["Volatilità"], 1),
                   "sharpe": _num(r["Sharpe"], 2), "dd": _pct(r["Massimo drawdown"], 1),
                   "extra": _pct(r["Extra-rendimento"], 2, True), "rot": _pct(r["Rotazione annua"], 0)}
                  for k, r in a.confronto_regimi.iterrows()]
    kpi = [
        ("Regime di oggi", dominante, f"probabilità {_pct(oggi.max(), 0)}"),
        ("Crisi nel prossimo mese", _pct(prev.iloc[-1], 0), "quota attesa del mese"),
        ("Sharpe per regimi", _num(rr["Sharpe"], 2), f"benchmark {_num(rb['Sharpe'], 2)}"),
        ("Extra-rendimento", _pct(rr["Extra-rendimento"], 2, True), "annuo, netto dei costi"),
        ("Massimo drawdown", _pct(rr["Massimo drawdown"], 1), f"benchmark {_pct(rb['Massimo drawdown'], 1)}"),
        ("Backtest ingenuo", _pct(t.loc[INGENUO, "Extra-rendimento"], 2, True), "extra-rendimento non replicabile"),
    ]
    ep = a.episodi
    meglio = [n for n in ep.index if ep.loc[n, REGIMI] > ep.loc[n, BENCH] + 1e-9]
    peggio = [n for n in ep.index if ep.loc[n, REGIMI] < ep.loc[n, BENCH] - 1e-9]
    pr = a.per_regime
    guadagno = [n for n in pr.index if pr.loc[n, REGIMI] > pr.loc[n, BENCH]]
    quota_ingenuo = (t.loc[INGENUO, "Extra-rendimento"] - rr["Extra-rendimento"]) / t.loc[INGENUO, "Extra-rendimento"] \
        if t.loc[INGENUO, "Extra-rendimento"] > 0 else None
    return {
        "cliente": c.cliente, "capitale": _eur(c.capitale), "oggi": date.today().strftime("%d/%m/%Y"),
        "dati_al": f"{a.rendimenti.index[-1]:%d/%m/%Y}", "dati_dal": f"{a.rendimenti.index[0]:%d/%m/%Y}",
        "test0": f"{test0:%m/%Y}", "test1": f"{test1:%m/%Y}", "mesi_test": len(a.risultati[BENCH].rendimenti),
        "frequenza": "settimanali" if a.periodi_anno == 52 else "mensili", "kpi": kpi, "k": a.stima.n_regimi,
        "caratteristiche": p.caratteristiche, "attivi": attivi, "regimi": regimi, "transizioni": transizioni,
        "nomi": a.stima.nomi, "colonne": colonne, "strategie": strategie, "nomi_strategie": STRATEGIE,
        "episodi": episodi, "per_regime": per_regime, "regime_vero": a.veri is not None, "proposta": proposta,
        "dominante": dominante, "p_oggi": [(n, _pct(x, 0)) for n, x in oggi.items()],
        "p_mese": [(n, _pct(x, 0)) for n, x in prev.items()], "banda": _pct(p.banda, 0),
        "bande": [(k, _pct(v, 0)) for k, v in p.bande.items()], "costo": f"{p.costo * 1e4:.0f}",
        "avversione": _num(p.avversione, 1), "restringimento": _pct(p.restringimento, 0),
        "rifai": p.rifai_ogni, "orizzonte": _num(p.orizzonte_costi, 0), "soglia": _pct(p.soglia_crisi, 0),
        "rischiosi": p.rischiosi, "rifugi": p.rifugi,
        "bench_comp": [(i, _pct(x, 0)) for i, x in a.config.benchmark.items() if x > 0],
        "boot_p_extra": _pct(float((boot["Extra-rendimento"] > 0).mean()), 0),
        "boot_p_sharpe": _pct(float((boot["Differenza di Sharpe"] > 0).mean()), 0),
        "boot_ci_extra": (_pct(q.iloc[0, 0], 1, True), _pct(q.iloc[1, 0], 1, True)),
        "boot_ci_sharpe": (_num(q.iloc[0, 1], 2, True), _num(q.iloc[1, 1], 2, True)), "n_boot": len(boot),
        "sel_bic": int(sel["BIC"].idxmin()), "sel_aic": int(sel["AIC"].idxmin()),
        "selezione": [{"k": k, "ll": _num(r["Log-verosimiglianza"], 1), "par": int(r["Parametri"]),
                       "aic": _num(r["AIC"], 1), "bic": _num(r["BIC"], 1)} for k, r in sel.iterrows()],
        "ep_meglio": meglio, "ep_peggio": peggio, "n_episodi": len(ep), "regimi_guadagno": guadagno,
        "rob_costo_max": f"{fila['Costo'].iloc[-1] * 1e4:.0f}",
        "rob_extra_max": _pct(fila["Extra-rendimento"].iloc[-1], 1, True),
        "rob_base": _pct(float(base["Extra-rendimento"].iloc[0]), 2, True) if len(base) else "–",
        "soglia_costi": soglia_costi, "ric": ric, "conf_k": conf_k, "rb": rb, "rr": rr, "t": t,
        "quota_ingenuo": _pct(quota_ingenuo, 0) if quota_ingenuo is not None else None,
        "rot_regimi": _pct(rr["Rotazione annua"], 0), "costi_regimi": _pct(rr["Costi annui"], 2),
        "eur": _eur, "pct": _pct, "num": _num, "REGIMI": REGIMI, "BENCH": BENCH, "INGENUO": INGENUO, "MV": MV,
        "RISKOFF": RISKOFF,
    }


def build_report(a: Analisi, output: str | Path, data_note: str = "", offline: bool = False) -> Path:
    """Scrive il report HTML. Con ``offline=True`` include plotly.js nel file (circa 4 MB)."""
    config_plotly = {"displaylogo": False, "responsive": True}
    grafici = {n: f(a).to_html(full_html=False, include_plotlyjs=False, config=config_plotly)
               for n, f in _disponibili(a).items()}
    env = Environment(loader=PackageLoader("regime_allocation", "templates"), autoescape=select_autoescape())
    html = env.get_template("report.html.j2").render(
        **contesto(a), grafici=grafici, nota=data_note, plotly_inline=get_plotlyjs() if offline else None,
        plotly_cdn=PLOTLY_CDN, demo=a.config.rendimenti is None and a.veri is not None,
        liquidita=a.config.parametri.liquidita, attivo_dati=dati.ATTIVI)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
