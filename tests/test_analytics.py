"""Pruebas unitarias para el módulo de analítica técnica."""

import numpy as np
import pandas as pd
import pytest
from opt_pyFE import proyeccion, bandas_bollinger


@pytest.fixture
def synthetic_stock_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = 100.0 + np.linspace(0, 20, 60) + np.random.normal(0, 0.5, 60)
    return pd.DataFrame({"Close": prices}, index=dates)


def test_proyeccion(synthetic_stock_data):
    pendientes = proyeccion(
        tickers="TEST",
        window=20,
        plot=False,
        data=synthetic_stock_data,
    )
    assert len(pendientes) == 1
    ticker, slope = pendientes[0]
    assert ticker == "TEST"
    # Dado que la serie tiene tendencia ascendente, la pendiente debe ser positiva
    assert slope > 0


def test_bandas_bollinger(synthetic_stock_data):
    res = bandas_bollinger(
        tickers="TEST",
        window=20,
        num_std=2.0,
        plot=False,
        data=synthetic_stock_data,
    )
    assert "TEST" in res
    df = res["TEST"]
    assert "MA20" in df.columns
    assert "UpperBand" in df.columns
    assert "LowerBand" in df.columns
    # La banda superior debe ser mayor o igual a la inferior
    valid_rows = df.dropna()
    assert (valid_rows["UpperBand"] >= valid_rows["LowerBand"]).all()
