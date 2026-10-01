"""Módulo principal que expone y unifica las funciones de opt_pyFE."""

from .data import (
    descargar_datos,
    getdata,
    get_data,
    calcular_rendimientos_log,
)
from .analytics import (
    proyeccion,
    bandas_bollinger,
    bollinger_bands,
)
from .portfolio import (
    simular_portafolios,
    simulate_portfolios,
    encontrar_mejor_portafolio,
    encontrar_minima_varianza,
    mostrar_resultados,
    desempeno,
    portfolio_performance,
    ejecutar_analisis,
)
from .risk import (
    historical_var,
    historicalVar,
    historical_cvar,
    historicalCVar,
    monte_carlo_sim,
    MonteCarlo,
    mc_var,
    mcVaR,
    mc_cvar,
    mcCVaR,
    resumen_riesgo,
    resum,
)
from .sentiment import (
    analizar_sentimiento,
    analyze_sentiment,
)

__all__ = [
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
