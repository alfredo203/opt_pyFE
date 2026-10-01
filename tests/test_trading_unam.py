"""Pruebas generales para el paquete `trading_unam`."""

import trading_unam


def test_package_metadata():
    assert hasattr(trading_unam, "__version__")
    assert trading_unam.__version__ == "0.2.0"
    assert hasattr(trading_unam, "__author__")
