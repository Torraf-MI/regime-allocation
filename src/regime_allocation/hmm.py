"""Hidden Markov Model gaussiano scritto da zero: stima EM (Baum-Welch), filtro, smoothing, Viterbi.

Il modello: un regime nascosto s_t ∈ {0, …, K−1} segue una catena di Markov con matrice di transizione A
(A[i, j] = P(s_t = j | s_{t−1} = i)); dato il regime, le osservazioni x_t sono normali multivariate con media e
covarianza proprie del regime. Tutte le ricorsioni sono scalate (Rabiner, 1989) per evitare underflow.

La distinzione che conta in un backtest:

* **probabilità filtrate** P(s_t | x_1, …, x_t): usano solo il passato, sono quelle disponibili in tempo reale;
* **probabilità lisciate** P(s_t | x_1, …, x_T): usano anche il futuro, sono più nette ma non replicabili.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _log_normale(X: np.ndarray, media: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Logaritmo della densità normale multivariata per ogni riga di X."""
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - media).T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (z**2).sum(axis=0))


@dataclass
class ModelloHMM:
    """Parametri di un HMM gaussiano con covarianze piene."""

    iniziale: np.ndarray  # (K,)
    transizioni: np.ndarray  # (K, K), righe che sommano a 1
    medie: np.ndarray  # (K, d)
    covarianze: np.ndarray  # (K, d, d)
    loglik: float = float("nan")
    storia_loglik: list[float] = field(default_factory=list)
    convergenza: bool = False

    @property
    def n_regimi(self) -> int:
        return len(self.iniziale)

    @property
    def n_parametri(self) -> int:
        k, d = self.medie.shape
        return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2

    # ----------------------------------------------------------------- inferenza
    def log_emissioni(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.column_stack([_log_normale(X, m, c) for m, c in zip(self.medie, self.covarianze)])

    def _avanti(self, logb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ricorsione in avanti scalata: alpha normalizzate, fattori di scala, emissioni riscalate."""
        T, K = logb.shape
        m = logb.max(axis=1, keepdims=True)
        b = np.exp(logb - m)
        alpha = np.empty((T, K))
        c = np.empty(T)
        a = self.iniziale * b[0]
        for t in range(T):
            if t:
                a = (alpha[t - 1] @ self.transizioni) * b[t]
            c[t] = a.sum()
            if not c[t] > 0:  # osservazione impossibile per tutti i regimi: si riparte dalla previsione
                a = alpha[t - 1] @ self.transizioni if t else self.iniziale.copy()
                c[t] = 1e-300
                alpha[t] = a / a.sum()
                continue
            alpha[t] = a / c[t]
        return alpha, np.log(c) + m[:, 0], b

    def filtra(self, X: np.ndarray) -> np.ndarray:
        """Probabilità filtrate P(s_t | x_1..x_t), una riga per osservazione."""
        return self._avanti(self.log_emissioni(X))[0]

    def log_verosimiglianza(self, X: np.ndarray) -> float:
        return float(self._avanti(self.log_emissioni(X))[1].sum())

    def _avanti_indietro(self, X: np.ndarray):
        logb = self.log_emissioni(X)
        alpha, logc, b = self._avanti(logb)
        T, K = alpha.shape
        c = np.exp(logc - logb.max(axis=1))  # fattori di scala riferiti alle emissioni riscalate
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = self.transizioni @ (b[t + 1] * beta[t + 1]) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        with np.errstate(over="ignore", invalid="ignore"):
            xi = self.transizioni * (alpha[:-1].T @ (b[1:] * beta[1:] / c[1:, None])) if T > 1 else np.zeros((K, K))
        xi = np.nan_to_num(xi, nan=0.0, posinf=0.0)
        return gamma, xi, float(logc.sum())

    def liscia(self, X: np.ndarray) -> np.ndarray:
        """Probabilità lisciate P(s_t | x_1..x_T): usano anche le osservazioni successive a t."""
        return self._avanti_indietro(X)[0]

    def viterbi(self, X: np.ndarray) -> np.ndarray:
        """Sequenza di regimi più probabile nel suo insieme."""
        logb = self.log_emissioni(X)
        T, K = logb.shape
        with np.errstate(divide="ignore"):
            logA, logpi = np.log(self.transizioni), np.log(self.iniziale)
        delta = logpi + logb[0]
        indietro = np.zeros((T, K), dtype=int)
        for t in range(1, T):
            s = delta[:, None] + logA
            indietro[t] = s.argmax(axis=0)
            delta = s.max(axis=0) + logb[t]
        percorso = np.empty(T, dtype=int)
        percorso[-1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            percorso[t - 1] = indietro[t, percorso[t]]
        return percorso

    def prevedi(self, probabilita: np.ndarray, passi: int = 1) -> np.ndarray:
        """Probabilità dei regimi ``passi`` periodi dopo, dalle probabilità di oggi."""
        return probabilita @ np.linalg.matrix_power(self.transizioni, passi)

    def stazionaria(self) -> np.ndarray:
        """Distribuzione di lungo periodo della catena (autovettore sinistro di autovalore 1)."""
        K = self.n_regimi
        M = np.vstack([self.transizioni.T - np.eye(K), np.ones(K)])
        p = np.linalg.lstsq(M, np.r_[np.zeros(K), 1.0], rcond=None)[0]
        p = np.clip(p, 0, None)
        return p / p.sum()

    def durate_attese(self) -> np.ndarray:
        """Durata media di un episodio di ciascun regime, in periodi: 1 / (1 − A[i, i])."""
        return 1.0 / np.maximum(1.0 - np.diag(self.transizioni), 1e-12)

    def bic(self, X: np.ndarray) -> float:
        return -2 * self.log_verosimiglianza(X) + self.n_parametri * np.log(len(X))

    def aic(self, X: np.ndarray) -> float:
        return -2 * self.log_verosimiglianza(X) + 2 * self.n_parametri

    def simula(self, T: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Simula T osservazioni: restituisce (X, regimi)."""
        rng = np.random.default_rng(seed)
        K, d = self.medie.shape
        s = np.empty(T, dtype=int)
        s[0] = rng.choice(K, p=self.iniziale)
        for t in range(1, T):
            s[t] = rng.choice(K, p=self.transizioni[s[t - 1]])
        L = np.linalg.cholesky(self.covarianze)
        X = self.medie[s] + np.einsum("tij,tj->ti", L[s], rng.standard_normal((T, d)))
        return X, s

    def riordina(self, ordine) -> ModelloHMM:
        """Lo stesso modello con i regimi in un altro ordine."""
        o = np.asarray(ordine)
        return ModelloHMM(self.iniziale[o], self.transizioni[np.ix_(o, o)], self.medie[o], self.covarianze[o],
                          self.loglik, list(self.storia_loglik), self.convergenza)


# --------------------------------------------------------------------------- #
# Stima
# --------------------------------------------------------------------------- #
def _iniziale_kmeans(X: np.ndarray, K: int, rng: np.random.Generator) -> ModelloHMM:
    """Punto di partenza: k-means++ con poche iterazioni, covarianze dai gruppi, catena persistente."""
    T, d = X.shape
    centri = [X[rng.integers(T)]]
    for _ in range(1, K):
        dist = np.min([((X - c) ** 2).sum(axis=1) for c in centri], axis=0)
        centri.append(X[rng.choice(T, p=dist / dist.sum())])
    centri = np.array(centri)
    for _ in range(10):
        g = np.argmin(((X[:, None, :] - centri[None]) ** 2).sum(axis=2), axis=1)
        centri = np.array([X[g == k].mean(axis=0) if (g == k).sum() > d else centri[k] for k in range(K)])
    cov_tot = np.cov(X.T).reshape(d, d)
    cov = np.array([np.cov(X[g == k].T).reshape(d, d) if (g == k).sum() > d + 1 else cov_tot for k in range(K)])
    cov = 0.5 * cov + 0.5 * cov_tot
    A = np.full((K, K), 0.1 / max(K - 1, 1)) + np.eye(K) * (0.9 - 0.1 / max(K - 1, 1)) if K > 1 else np.ones((1, 1))
    return ModelloHMM(np.full(K, 1.0 / K), A, centri, cov)


def _passo_em(mod: ModelloHMM, X: np.ndarray, reg: np.ndarray, persistenza: float) -> tuple[ModelloHMM, float]:
    gamma, xi, ll = mod._avanti_indietro(X)
    K, d = mod.medie.shape
    nk = gamma.sum(axis=0) + 1e-10
    medie = (gamma.T @ X) / nk[:, None]
    cov = np.empty((K, d, d))
    for k in range(K):
        D = X - medie[k]
        cov[k] = (gamma[:, k, None] * D).T @ D / nk[k] + reg
    A = xi + persistenza * np.eye(K)  # pseudo-osservazioni sulla diagonale (prior di Dirichlet)
    somme = A.sum(axis=1, keepdims=True)
    A = np.where(somme > 0, A / np.where(somme > 0, somme, 1.0), mod.transizioni)
    pi = np.clip(gamma[0], 1e-6, None)
    return ModelloHMM(pi / pi.sum(), A, medie, cov), ll


def stima_hmm(X, n_regimi: int = 3, n_avvii: int = 6, max_iter: int = 500, tol: float = 1e-6,
              persistenza: float = 0.0, seed: int = 0, iniziale: ModelloHMM | None = None) -> ModelloHMM:
    """Stima di massima verosimiglianza con l'algoritmo EM (Baum-Welch) da più punti di partenza.

    ``persistenza`` aggiunge pseudo-transizioni sulla diagonale di A (prior di Dirichlet): rende i regimi più
    stabili e riduce i falsi cambi di regime. ``iniziale`` permette di ripartire da una stima precedente (utile
    nel walk-forward); in quel caso gli avvii casuali si aggiungono a quel punto di partenza.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    T, d = X.shape
    if T < n_regimi * (d + 2):
        raise ValueError(f"Troppe poche osservazioni ({T}) per {n_regimi} regimi su {d} variabili")
    rng = np.random.default_rng(seed)
    reg = np.eye(d) * 1e-4 * np.var(X, axis=0).mean()
    partenze = [iniziale] if iniziale is not None else []
    partenze += [_iniziale_kmeans(X, n_regimi, rng) for _ in range(n_avvii)]
    migliore = None
    for mod in partenze:
        storia = []
        conv = False
        try:
            for _ in range(max_iter):
                mod, ll = _passo_em(mod, X, reg, persistenza)
                storia.append(ll)
                if len(storia) > 1 and abs(storia[-1] - storia[-2]) < tol * (1 + abs(storia[-2])):
                    conv = True
                    break
        except np.linalg.LinAlgError:
            continue
        mod.loglik = mod.log_verosimiglianza(X)
        mod.storia_loglik, mod.convergenza = storia, conv
        if migliore is None or mod.loglik > migliore.loglik:
            migliore = mod
    if migliore is None:
        raise RuntimeError("Stima HMM non riuscita da nessun punto di partenza")
    return migliore


def selezione_regimi(X, massimo: int = 5, **kw) -> list[dict]:
    """Confronta modelli da 1 a ``massimo`` regimi con log-verosimiglianza, AIC e BIC."""
    out = []
    for k in range(1, massimo + 1):
        m = stima_hmm(X, k, **kw)
        out.append({"Regimi": k, "Log-verosimiglianza": m.loglik, "Parametri": m.n_parametri,
                    "AIC": m.aic(X), "BIC": m.bic(X)})
    return out
