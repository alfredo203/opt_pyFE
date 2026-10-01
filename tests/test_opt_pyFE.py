"""Pruebas de retrocompatibilidad para el alias `opt_pyFE`."""

import opt_pyFE
import trading_unam


def test_package_metadata_and_backward_compatibility():
    assert hasattr(opt_pyFE, "__version__")
    assert opt_pyFE.__version__ == trading_unam.__version__
    assert hasattr(opt_pyFE, "__author__")
    assert opt_pyFE.descargar_datos is trading_unam.descargar_datos
    assert opt_pyFE.simular_portafolios is trading_unam.simular_portafolios
