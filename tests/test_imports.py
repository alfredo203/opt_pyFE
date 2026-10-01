"""Pruebas de integridad de importaciones y retrocompatibilidad de nombres."""

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
        assert hasattr(opt_pyFE, fn_name), f"Falta la función {fn_name} en la API pública"


def test_alias_equivalence():
    assert opt_pyFE.getdata is opt_pyFE.get_data
    assert opt_pyFE.bandas_bollinger is opt_pyFE.bollinger_bands
    assert opt_pyFE.simular_portafolios is opt_pyFE.simulate_portfolios
    assert opt_pyFE.desempeno is opt_pyFE.portfolio_performance
    assert opt_pyFE.historicalVar is opt_pyFE.historical_var
    assert opt_pyFE.historicalCVar is opt_pyFE.historical_cvar
    assert opt_pyFE.MonteCarlo is opt_pyFE.monte_carlo_sim
    assert opt_pyFE.mcVaR is opt_pyFE.mc_var
    assert opt_pyFE.mcCVaR is opt_pyFE.mc_cvar
    assert opt_pyFE.resum is opt_pyFE.resumen_riesgo
    assert opt_pyFE.analizar_sentimiento is opt_pyFE.analyze_sentiment
