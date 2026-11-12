import json

import numpy as np
import pandas as pd
import pytest

from regime_allocation import cli, configurazione, dati
from regime_allocation.analisi import BENCH, INGENUO, REGIMI, STRATEGIE
from regime_allocation.report import build_report, contesto


def test_regimi_riconosciuti(analisi_demo):
    a = analisi_demo
    assert a.stima.nomi == ["Espansione", "Tassi in rialzo", "Crisi"]
    r = a.riconoscimento
    assert r["accuratezza"] > 0.8
    assert set(r["confusione"].index) == set(dati.REGIMI)
    # le crisi fanno perdere le azioni e guadagnare i titoli di Stato; quando salgono i tassi scendono entrambi
    t = a.stima.tabella().set_index(["Regime", "Attività"])["Rendimento"]
    assert t["Crisi", "Azioni globali"] < 0 < t["Crisi", "Titoli di Stato euro"]
    assert t["Tassi in rialzo", "Titoli di Stato euro"] < 0 and t["Tassi in rialzo", "Azioni globali"] < 0
    corr = a.stima.correlazione("Azioni globali", "Titoli di Stato euro")
    assert corr["Crisi"] < 0 < corr["Tassi in rialzo"]
    assert int(a.selezione["BIC"].idxmin()) == 3


def test_backtest_e_confronti(analisi_demo):
    a = analisi_demo
    assert list(a.tabella.index) == STRATEGIE
    n = {len(r.rendimenti) for r in a.risultati.values()}
    assert len(n) == 1
    t = a.tabella
    assert t.loc[BENCH, "Extra-rendimento"] == 0
    assert t.loc[REGIMI, "Sharpe"] > t.loc[BENCH, "Sharpe"]
    assert t.loc[INGENUO, "Sharpe"] > t.loc[REGIMI, "Sharpe"]  # il senno di poi esagera
    lo = dati.BENCHMARK - a.config.parametri.bande_per(dati.BENCHMARK.index) - 1e-9
    hi = dati.BENCHMARK + a.config.parametri.bande_per(dati.BENCHMARK.index) + 1e-9
    w = a.risultati[REGIMI].pesi
    assert np.allclose(w.sum(axis=1), 1) and (w >= lo.clip(lower=0)).all().all() and (w <= hi).all().all()
    assert set(a.episodi.index) == {e[0] for e in dati.EPISODI}
    assert len(a.robustezza) == 30


def test_proposta(analisi_demo):
    p = analisi_demo.proposta
    assert p["Per regimi"].sum() == pytest.approx(1)
    assert p["Operazione"].sum() == pytest.approx(0, abs=1e-6)
    assert p["Importo"].sum() == pytest.approx(analisi_demo.config.capitale)
    assert analisi_demo.probabilita_mese.sum() == pytest.approx(1)


def test_report(analisi_demo, tmp_path):
    out = build_report(analisi_demo, tmp_path / "r.html", data_note="Nota di prova")
    html = out.read_text(encoding="utf-8")
    assert html.count('class="plotly-graph-div"') == 11
    assert "Famiglia Conti" in html and "Nota di prova" in html and "Fortuna o capacità" in html
    c = contesto(analisi_demo)
    assert c["sel_bic"] == 3 and len(c["strategie"]) == 5


def test_csv_settimanale_e_mensile(rendimenti, tmp_path):
    veri = dati.regimi_veri()
    dati.salva_csv(rendimenti, tmp_path / "s.csv", veri)
    r, v = dati.carica_csv(tmp_path / "s.csv")
    assert np.allclose(r.values, rendimenti.values, atol=1e-6) and (v.values == veri.values).all()
    assert dati.periodi_per_anno(r.index) == 52
    m = dati.mensili(rendimenti) * 100  # in percentuale: vengono riconosciuti e convertiti
    m.index = m.index.strftime("%Y-%m")
    m.index.name = "data"
    m.to_csv(tmp_path / "m.csv")
    rm, vm = dati.carica_csv(tmp_path / "m.csv")
    assert vm is None and dati.periodi_per_anno(rm.index) == 12
    assert rm.iloc[0, 0] == pytest.approx(dati.mensili(rendimenti).iloc[0, 0], abs=1e-6)
    pd.DataFrame({"data": ["2020-01-03"], "A": [0.1]}).to_csv(tmp_path / "corto.csv", index=False)
    with pytest.raises(ValueError):
        dati.carica_csv(tmp_path / "corto.csv")


@pytest.mark.parametrize("cattiva", [{}, {"benchmark": {"A": 0.5}}, {"benchmark": {"A": 1.2, "B": -0.2}},
                                     {"benchmark": {"A": 1.0}, "banda": 2}, {"benchmark": {"A": 1.0}, "regimi": 9}])
def test_configurazioni_non_valide(cattiva):
    with pytest.raises(ValueError):
        configurazione.da_dizionario(cattiva)


def test_cli_con_configurazione(rendimenti, tmp_path, capsys):
    dati.salva_csv(dati.mensili(rendimenti.loc[:"2016-12-31"]), tmp_path / "dati.csv")
    cfg = {"cliente": "Famiglia Verdi", "capitale": 250000, "rendimenti": "dati.csv",
           "benchmark": {"Azioni globali": 0.5, "Titoli di Stato euro": 0.4, "Liquidità": 0.1},
           "caratteristiche": ["Azioni globali", "Titoli di Stato euro"], "regimi": 2, "inizio_test": "2006-01",
           "rifai_ogni": 24, "banda": 0.1, "bootstrap": 100, "confronto_regimi": False}
    (tmp_path / "c.json").write_text(json.dumps(cfg), encoding="utf-8")
    out = tmp_path / "report.html"
    assert cli.main(["--config", str(tmp_path / "c.json"), "--output", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert "Famiglia Verdi" in html and "dati mensili" in html and "Quanto bene li riconosce" not in html
    testo = capsys.readouterr().out
    assert "Report salvato" in testo and "Espansione" in testo and "Crisi" in testo
