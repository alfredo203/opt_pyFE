"""Pruebas unitarias para el módulo de medición de riesgo."""

import numpy as np
import pandas as pd
import pytest
from trading_unam import (
    historical_var,
    historical_cvar,
    monte_carlo_sim,
    mc_var,
    mc_cvar,
    resumen_riesgo,
)


def test_historical_var_and_cvar():
    np.random.seed(42)
    # Serie de rendimientos con media 0 y desviación 0.02
    returns = pd.Series(np.random.normal(0, 0.02, 1000))

    var_5 = historical_var(returns, alpha=5.0)
    cvar_5 = historical_cvar(returns, alpha=5.0)

    # El cuantil 5% debe ser negativo en una distribución normal centrada en 0
    assert var_5 < 0
    # Por definición de cola de pérdidas, CVaR <= VaR (la pérdida promedio en la cola es más severa o igual)
    assert cvar_5 <= var_5


def test_monte_carlo_sim():
    np.random.seed(42)
    mu = np.array([0.0005, 0.0008])
    peso = np.array([0.6, 0.4])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    initial_portfolio = 10000.0

    sims, results = monte_carlo_sim(
        mc_sims=50,
        T=20,
        media_rendimiento=mu,
        peso=peso,
        initialPortfolio=initial_portfolio,
        covmatrix=cov,
        plot=False,
    )

    assert sims.shape == (20, 50)
    assert len(results) == 50
    assert (results > 0).all()

    mc_v = mc_var(results, alpha=5.0)
    mc_cv = mc_cvar(results, alpha=5.0)
    assert mc_cv <= mc_v


def test_resumen_riesgo(capsys):
    res = resumen_riesgo(
        inversion_inicial=10000,
        initialPortfolio=10000,
        hVaR=0.05,
        hCVaR=0.07,
        MCVaR=550,
        MCCVaR=750,
        pRet=0.08,
    )

    assert res["historical_VaR_monto"] == 500.0
    assert res["historical_CVaR_monto"] == 700.0
    assert res["MC_VaR_monto"] == 550.0
    assert res["MC_CVaR_monto"] == 750.0
    assert res["initial_portfolio"] == 10000.0
    assert res["portfolio_performance_monto"] == 800.0

    captured = capsys.readouterr()
    assert "RESUMEN DE RIESGO" in captured.out
