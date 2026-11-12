import numpy as np
import pandas as pd
import pytest

from regime_allocation.allocazione import (
    ancora,
    limiti,
    momenti_misti,
    ottimizza,
    regola_risk_off,
    rendimenti_equilibrio,
)


@pytest.fixture
def mercato():
    vol = np.array([0.16, 0.05, 0.15, 0.0])
    C = np.array([[1, -0.2, 0.1, 0], [-0.2, 1, 0.2, 0], [0.1, 0.2, 1, 0], [0, 0, 0, 1]])
    S = C * np.outer(vol, vol)
    b = np.array([0.4, 0.4, 0.1, 0.1])
    return S, b


def test_momenti_della_mistura():
    rng = np.random.default_rng(0)
    medie = np.array([[0.01, 0.0], [-0.03, 0.01]])
    cov = np.array([np.diag([0.02, 0.01]) ** 2, np.array([[0.05, -0.01], [-0.01, 0.02]]) ** 2])
    p = np.array([0.7, 0.3])
    mu, S = momenti_misti(p, medie, cov)
    k = rng.choice(2, size=400_000, p=p)
    x = np.vstack([rng.multivariate_normal(medie[j], cov[j], size=(k == j).sum()) for j in range(2)])
    assert np.allclose(mu, x.mean(axis=0), atol=2e-4)
    assert np.allclose(S, np.cov(x.T), rtol=0.03)
    mu1, S1 = momenti_misti(np.array([1.0, 0.0]), medie, cov)
    assert np.allclose(mu1, medie[0]) and np.allclose(S1, cov[0])


def test_equilibrio_restituisce_il_benchmark(mercato):
    S, b = mercato
    pi = rendimenti_equilibrio(S, b, 4.0)
    assert pi[-1] == 0  # la liquidità non ha premio
    w = ottimizza(pi, S, b, None, 4.0, banda=0.3, costo=0.0)
    assert np.allclose(w, b, atol=1e-4)
    # con regimi alle probabilità di lungo periodo le medie ancorate tornano all'equilibrio
    medie = np.array([[0.1, 0.0, 0.05, 0], [-0.3, 0.06, 0.1, 0]])
    tot = np.array([0.7, 0.3]) @ medie
    assert np.allclose(np.array([0.7, 0.3]) @ ancora(medie, tot, pi, 0.5), pi)


def test_ottimizzazione_rispetta_le_bande(mercato):
    S, b = mercato
    mu = np.array([0.30, -0.10, 0.0, 0.0])
    w = ottimizza(mu, S, b, None, 2.0, banda=0.1)
    lo, hi = limiti(b, 0.1)
    assert w.sum() == pytest.approx(1) and (w >= lo - 1e-9).all() and (w <= hi + 1e-9).all()
    assert w[0] == pytest.approx(0.5, abs=1e-6) and w[1] == pytest.approx(0.3, abs=1e-6)
    w2 = ottimizza(mu, S, b, None, 2.0, banda=np.array([0.05, 0.1, 0.1, 0.1]))
    assert w2[0] == pytest.approx(0.45, abs=1e-6)


def test_soluzione_interna_uguale_alla_forma_chiusa(mercato):
    S, _ = mercato
    S, b = S[:3, :3], np.array([0.4, 0.4, 0.2])
    mu = np.array([0.04, 0.01, 0.02])
    lam = 6.0
    w = ottimizza(mu, S, b, None, lam, banda=1.0)
    Si = np.linalg.inv(S)
    g = (np.ones(3) @ Si @ mu - lam) / (np.ones(3) @ Si @ np.ones(3))
    atteso = Si @ (mu - g) / lam
    assert np.allclose(w, atteso, atol=1e-5)


def test_costi_alti_bloccano_le_operazioni(mercato):
    S, b = mercato
    mu = np.array([0.05, 0.0, 0.01, 0.0])
    w0 = np.array([0.35, 0.45, 0.1, 0.1])
    assert np.allclose(ottimizza(mu, S, b, w0, 4.0, 0.1, costo=0.05), w0, atol=1e-6)
    assert not np.allclose(ottimizza(mu, S, b, w0, 4.0, 0.1, costo=0.0), w0, atol=1e-3)


def test_regola_risk_off():
    b = pd.Series({"Azioni": 0.4, "Stato": 0.3, "Societarie": 0.15, "Liquidità": 0.15})
    assert regola_risk_off(b, 0.4, 0.5, 0.1, ["Azioni", "Societarie"], ["Stato", "Liquidità"]).equals(b)
    w = regola_risk_off(b, 0.8, 0.5, 0.1, ["Azioni", "Societarie"], ["Stato", "Liquidità"])
    assert w.sum() == pytest.approx(1)
    assert w["Azioni"] == pytest.approx(0.3) and w["Societarie"] == pytest.approx(0.05)
    assert w["Stato"] == pytest.approx(0.4) and w["Liquidità"] == pytest.approx(0.25)
