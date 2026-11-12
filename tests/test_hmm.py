import itertools

import numpy as np
import pytest

from regime_allocation.hmm import ModelloHMM, selezione_regimi, stima_hmm


def modello(k=2):
    if k == 2:
        A = np.array([[0.95, 0.05], [0.10, 0.90]])
        medie = np.array([[0.5, 0.0], [-1.0, 0.8]])
        cov = np.array([[[0.5, 0.1], [0.1, 0.4]], [[2.0, -0.6], [-0.6, 1.0]]])
    else:
        A = np.array([[0.95, 0.03, 0.02], [0.10, 0.88, 0.02], [0.05, 0.05, 0.90]])
        medie = np.array([[0.4, 0.1], [-0.6, -0.8], [-1.2, 0.9]])
        cov = np.array([[[0.4, 0.0], [0.0, 0.3]], [[1.0, 0.5], [0.5, 1.0]], [[2.5, -0.8], [-0.8, 1.2]]])
    return ModelloHMM(np.full(k, 1 / k), A, medie, cov)


def densita(m, x):
    return np.exp(m.log_emissioni(x[None, :])[0])


def forza_bruta(m, X):
    """Probabilità congiunte di tutte le sequenze di regimi: filtro, smoothing e Viterbi esatti."""
    T, K = len(X), m.n_regimi
    seq = list(itertools.product(range(K), repeat=T))
    b = np.array([densita(m, x) for x in X])
    p = np.array([m.iniziale[s[0]] * b[0, s[0]] * np.prod([m.transizioni[s[t - 1], s[t]] * b[t, s[t]]
                                                           for t in range(1, T)]) for s in seq])
    liscia = np.array([[sum(p[i] for i, s in enumerate(seq) if s[t] == k) for k in range(K)] for t in range(T)])
    filtra = []
    for t in range(T):
        sub = list(itertools.product(range(K), repeat=t + 1))
        q = np.array([m.iniziale[s[0]] * b[0, s[0]] * np.prod([m.transizioni[s[u - 1], s[u]] * b[u, s[u]]
                                                               for u in range(1, t + 1)]) for s in sub])
        filtra.append([sum(q[i] for i, s in enumerate(sub) if s[-1] == k) for k in range(K)])
    filtra = np.array(filtra)
    return (filtra / filtra.sum(axis=1, keepdims=True), liscia / liscia.sum(axis=1, keepdims=True),
            np.array(seq[int(p.argmax())]), np.log(p.sum()))


def test_ricorsioni_uguali_alla_forza_bruta():
    m = modello(2)
    X, _ = m.simula(7, seed=3)
    f, s, v, ll = forza_bruta(m, X)
    assert np.allclose(m.filtra(X), f, atol=1e-10)
    assert np.allclose(m.liscia(X), s, atol=1e-10)
    assert (m.viterbi(X) == v).all()
    assert m.log_verosimiglianza(X) == pytest.approx(ll, abs=1e-9)


def test_filtro_non_guarda_il_futuro():
    m = modello(3)
    X, _ = m.simula(300, seed=1)
    f = m.filtra(X)
    X2 = X.copy()
    X2[200:] = 5.0  # futuro completamente diverso
    assert np.allclose(m.filtra(X2)[:200], f[:200])
    assert np.allclose(m.filtra(X[:120]), f[:120])
    assert not np.allclose(m.liscia(X2)[:200], m.liscia(X)[:200])  # lo smoothing invece cambia
    assert np.allclose(m.liscia(X)[-1], f[-1])  # all'ultima data coincidono
    assert np.allclose(f.sum(axis=1), 1)


def test_stima_ritrova_i_parametri():
    vero = modello(2)
    X, s = vero.simula(4000, seed=7)
    m = stima_hmm(X, 2, n_avvii=3, seed=0)
    m = m.riordina(np.argsort(m.covarianze[:, 0, 0]))
    assert np.allclose(m.medie, vero.medie, atol=0.1)
    assert np.allclose(m.covarianze, vero.covarianze, atol=0.2)
    assert np.allclose(m.transizioni, vero.transizioni, atol=0.03)
    assert (m.viterbi(X) == s).mean() > 0.95
    assert m.convergenza


def test_em_non_diminuisce_la_verosimiglianza():
    X, _ = modello(3).simula(800, seed=2)
    m = stima_hmm(X, 3, n_avvii=1, seed=4)
    assert (np.diff(m.storia_loglik) > -1e-7).all()


def test_bic_sceglie_il_numero_giusto_di_regimi():
    X, _ = modello(3).simula(2500, seed=5)
    sel = selezione_regimi(X, 4, n_avvii=3)
    assert min(sel, key=lambda r: r["BIC"])["Regimi"] == 3
    assert [r["Parametri"] for r in sel] == [5, 13, 23, 35]


def test_stazionaria_durate_e_previsione():
    m = modello(2)
    p = m.stazionaria()
    assert np.allclose(p @ m.transizioni, p) and p.sum() == pytest.approx(1)
    assert np.allclose(p, [2 / 3, 1 / 3])
    assert np.allclose(m.durate_attese(), [20, 10])
    assert np.allclose(m.prevedi(np.array([1.0, 0.0]), 500), p)
    r = m.riordina([1, 0])
    assert np.allclose(r.transizioni, [[0.9, 0.1], [0.05, 0.95]]) and np.allclose(r.medie[0], m.medie[1])


def test_troppe_poche_osservazioni():
    with pytest.raises(ValueError):
        stima_hmm(np.zeros((5, 2)), 3)
