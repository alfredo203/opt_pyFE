"""Pruebas unitarias para el módulo de optimización y simulación de portafolios."""

import numpy as np
import pandas as pd
import pytest
from opt_pyFE import (
    simular_portafolios,
    encontrar_mejor_portafolio,
    encontrar_minima_varianza,
    desempeno,
)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    data = np.random.normal(0.001, 0.02, size=(100, 3))
    return pd.DataFrame(data, columns=["ACT1", "ACT2", "ACT3"])


def test_simular_portafolios(sample_returns):
    num_sims = 500
    weights, ret_esp, vol_esp, sharpe = simular_portafolios(sample_returns, num_portafolios=num_sims)

    assert weights.shape == (num_sims, 3)
    assert len(ret_esp) == num_sims
    assert len(vol_esp) == num_sims
    assert len(sharpe) == num_sims

    # Comprobar que los pesos suman aproximadamente 1
    assert np.allclose(weights.sum(axis=1), 1.0)
    # Volatilidad estrictamente positiva
    assert (vol_esp >= 0).all()


def test_encontrar_mejor_portafolio(sample_returns):
    weights, ret_esp, vol_esp, sharpe = simular_portafolios(sample_returns, num_portafolios=200)
    best_w, best_r, best_v, max_s = encontrar_mejor_portafolio(weights, ret_esp, vol_esp, sharpe)

    assert len(best_w) == 3
    assert np.isclose(best_w.sum(), 1.0)
    assert max_s == sharpe.max()


def test_encontrar_minima_varianza(sample_returns):
    weights, ret_esp, vol_esp, sharpe = simular_portafolios(sample_returns, num_portafolios=200)
    min_w, min_r, min_v, min_s = encontrar_minima_varianza(weights, ret_esp, vol_esp, sharpe)

    assert len(min_w) == 3
    assert min_v == vol_esp.min()


def test_desempeno():
    peso = np.array([0.5, 0.5])
    mu = np.array([0.01, 0.02])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    time = 10

    rend, std = desempeno(peso, mu, cov, time=time)

    expected_rend = (0.5 * 0.01 + 0.5 * 0.02) * 10
    assert np.isclose(rend, expected_rend)
    assert std > 0
