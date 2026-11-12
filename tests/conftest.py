import pytest

from regime_allocation import configurazione, dati
from regime_allocation.analisi import analizza


@pytest.fixture(scope="session")
def rendimenti():
    return dati.rendimenti_dimostrativi()


@pytest.fixture(scope="session")
def analisi_demo(rendimenti):
    cfg = configurazione.esempio()
    cfg.confronto_regimi = False
    cfg.n_bootstrap = 300
    cfg.parametri.rifai_ogni = 24
    return analizza(cfg, rendimenti, dati.regimi_veri())
