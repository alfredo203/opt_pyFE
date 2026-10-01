"""Pruebas unitarias para el módulo de datos y rendimientos."""

import numpy as np
import pandas as pd
import pytest
from opt_pyFE import calcular_rendimientos_log


def test_calcular_rendimientos_log():
    # Precios sintéticos
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df_precios = pd.DataFrame(
        {
            "AAPL": [100.0, 105.0, 102.0, 110.0, 115.0],
            "MSFT": [200.0, 202.0, 204.0, 201.0, 210.0],
        },
        index=dates,
    )

    log_ret = calcular_rendimientos_log(df_precios)

    assert isinstance(log_ret, pd.DataFrame)
    assert len(log_ret) == 4
    assert "AAPL" in log_ret.columns
    assert "MSFT" in log_ret.columns

    # Validar cálculo manual: ln(105/100)
    expected_first_aapl = np.log(105.0 / 100.0)
    assert np.isclose(log_ret["AAPL"].iloc[0], expected_first_aapl)
