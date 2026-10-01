"""Pruebas generales para el paquete `opt_pyFE`."""

import opt_pyFE


def test_package_metadata():
    assert hasattr(opt_pyFE, "__version__")
    assert opt_pyFE.__version__ == "0.2.0"
    assert hasattr(opt_pyFE, "__author__")
