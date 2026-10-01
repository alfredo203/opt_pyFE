"""Pruebas de integridad de importaciones y retrocompatibilidad de nombres."""

import trading_unam
import opt_pyFE


def test_public_api_exports():
    expected_functions = [
        # Data
        "descargar_datos",
        "getdata",
        "get_data",
        "calcular_rendimientos_log",
        # Analytics
        "proyeccion",
        "bandas_bollinger",
        "bollinger_bands",
        # Portfolio
        "simular_portafolios",
        "simulate_portfolios",
        "encontrar_mejor_portafolio",
        "encontrar_minima_varianza",
        "mostrar_resultados",
        "desempeno",
        "portfolio_performance",
        "ejecutar_analisis",
        # Risk
        "historical_var",
        "historicalVar",
        "historical_cvar",
        "historicalCVar",
        "monte_carlo_sim",
        "MonteCarlo",
        "mc_var",
        "mcVaR",
        "mc_cvar",
        "mcCVaR",
        "resumen_riesgo",
        "resum",
        # Sentiment
        "analizar_sentimiento",
        "analyze_sentiment",
    ]

    for fn_name in expected_functions:
        assert hasattr(trading_unam, fn_name), f"Falta la función {fn_name} en la API pública de trading_unam"
        assert hasattr(opt_pyFE, fn_name), f"Falta la función {fn_name} en el alias opt_pyFE"


def test_alias_equivalence():
    assert trading_unam.getdata is trading_unam.get_data
    assert trading_unam.bandas_bollinger is trading_unam.bollinger_bands
    assert trading_unam.simular_portafolios is trading_unam.simulate_portfolios
    assert trading_unam.desempeno is trading_unam.portfolio_performance
    assert trading_unam.historicalVar is trading_unam.historical_var
    assert trading_unam.historicalCVar is trading_unam.historical_cvar
    assert trading_unam.MonteCarlo is trading_unam.monte_carlo_sim
    assert trading_unam.mcVaR is trading_unam.mc_var
    assert trading_unam.mcCVaR is trading_unam.mc_cvar
    assert trading_unam.resum is trading_unam.resumen_riesgo
    assert trading_unam.analizar_sentimiento is trading_unam.analyze_sentiment
