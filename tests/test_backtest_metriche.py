import numpy as np
import pandas as pd
import pytest

from regime_allocation import dati
from regime_allocation.backtest import Parametri, simula, walk_forward
from regime_allocation.configurazione import CARATTERISTICHE, RIFUGI, RISCHIOSI
from regime_allocation.metriche import bootstrap_differenza, cagr, drawdown, indici_bootstrap, metriche


def parametri(**kw):
    return Parametri(list(CARATTERISTICHE), dati.LIQUIDITA, rischiosi=list(RISCHIOSI), rifugi=list(RIFUGI), **kw)


def test_benchmark_calcolato_a_mano(rendimenti):
    par = parametri(inizio_test="2020-01", costo=0.002)
    r = simula("B", "benchmark", None, rendimenti, dati.BENCHMARK, par)
    M = dati.mensili(rendimenti)[dati.BENCHMARK.index]
    b = dati.BENCHMARK.values
    w = b.copy()
    attesi = []
    for i, d in enumerate(r.pesi.index):
        x = M.values[M.index.get_loc(d) + 1]
        lordo = b @ x
        attesi.append(lordo - (0.002 * np.abs(b - w).sum() if i else 0.0))
        w = b * (1 + x) / (1 + lordo)
    assert np.allclose(r.rendimenti.values, attesi)
    assert r.rendimenti.index[0] == pd.Timestamp("2020-01-31")
    assert (r.rotazione.iloc[1:] > 0).all()


def test_mensili_composti(rendimenti):
    m = dati.mensili(rendimenti)
    gen = rendimenti.loc["2008-10"]
    assert m.loc["2008-10-31"].values == pytest.approx((1 + gen).prod().values - 1)
    assert len(m) == 322


def test_walk_forward_senza_dati_futuri(rendimenti):
    par = parametri(inizio_test="2008-01", rifai_ogni=6, n_avvii=2)
    corti = rendimenti.loc[:"2010-06-30"]
    a = walk_forward(corti, par)
    lunghi = rendimenti.loc[:"2011-06-30"].copy()
    lunghi.loc["2010-07-01":] *= -3  # un futuro molto diverso non cambia le decisioni passate
    b = walk_forward(lunghi, par)
    n = len(a.date)
    assert (b.date[:n] == a.date).all()
    assert np.allclose(b.prossimo[:n], a.prossimo)
    assert np.allclose(np.array(b.medie[:n]), np.array(a.medie))
    assert np.allclose(b.settimanali.loc[a.settimanali.index].values, a.settimanali.values)


def test_metriche_note():
    idx = pd.period_range("2020-01", periods=12, freq="M").to_timestamp(how="end")
    r = pd.Series([0.1, -0.2, 0.05, 0.05] + [0.0] * 8, index=idx)
    assert drawdown(r).min() == pytest.approx(-0.2)
    assert cagr(pd.Series([0.01] * 12, index=idx)) == pytest.approx(1.01**12 - 1)
    rf = pd.Series(0.0, index=idx)
    m = metriche(r, rf, benchmark=r)
    assert m["Massimo drawdown"] == pytest.approx(-0.2) and m["Mese peggiore"] == pytest.approx(-0.2)
    assert m["Extra-rendimento"] == 0 and m["Tracking error"] == 0


def test_bootstrap():
    rng = np.random.default_rng(1)
    i = indici_bootstrap(200, 6, rng)
    assert i.min() >= 0 and i.max() < 200
    assert 3 < 200 / (1 + (np.diff(i) != 1).sum()) < 12  # blocchi lunghi in media circa 6
    idx = pd.period_range("2000-01", periods=240, freq="M").to_timestamp(how="end")
    b = pd.Series(rng.normal(0.005, 0.03, 240), index=idx)
    rf = pd.Series(0.001, index=idx)
    d = bootstrap_differenza(b + 0.002, b, rf, n=300)
    assert (d["Extra-rendimento"] > 0).all() and (d["Differenza di Sharpe"] > 0).all()
    assert bootstrap_differenza(b, b, rf, n=50).abs().max().max() < 1e-12
