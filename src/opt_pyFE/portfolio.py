"""Módulo para simulación, optimización y análisis de desempeño de portafolios."""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from .data import descargar_datos, calcular_rendimientos_log


def simular_portafolios(
    log_returns: pd.DataFrame,
    num_portafolios: int = 5000,
    risk_free_rate: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genera simulaciones de carteras con pesos aleatorios bajo la teoría moderna de portafolios.

    Parameters
    ----------
    log_returns : pd.DataFrame
        DataFrame con los rendimientos continuos de los activos (cada columna es un activo).
    num_portafolios : int, default=5000
        Número de portafolios aleatorios a generar.
    risk_free_rate : float, default=0.0
        Tasa libre de riesgo para el cálculo del Ratio de Sharpe.

    Returns
    -------
    weight : np.ndarray
        Matriz de forma (num_portafolios, num_activos) con las ponderaciones normalizadas a 1.
    ReturnEsp : np.ndarray
        Vector de rendimientos esperados por portafolio.
    VolEsp : np.ndarray
        Vector de volatilidades (desviaciones estándar) esperadas por portafolio.
    RadioSharpe : np.ndarray
        Vector con el Ratio de Sharpe de cada portafolio.
    """
    num_activos = log_returns.shape[1]
    weight = np.zeros((num_portafolios, num_activos))
    ReturnEsp = np.zeros(num_portafolios)
    VolEsp = np.zeros(num_portafolios)
    RadioSharpe = np.zeros(num_portafolios)

    meanlogReturns = log_returns.mean().values
    Sigma = log_returns.cov().values

    for k in range(num_portafolios):
        w = np.random.random(num_activos)
        w /= np.sum(w)
        weight[k, :] = w

        ReturnEsp[k] = np.sum(meanlogReturns * w)
        VolEsp[k] = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        RadioSharpe[k] = (ReturnEsp[k] - risk_free_rate) / (VolEsp[k] if VolEsp[k] > 0 else 1e-8)

    return weight, ReturnEsp, VolEsp, RadioSharpe


# Alias amigable con PEP 8
simulate_portfolios = simular_portafolios


def encontrar_mejor_portafolio(
    weight: np.ndarray,
    ReturnEsp: np.ndarray,
    VolEsp: np.ndarray,
    RadioSharpe: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """Identifica el portafolio que maximiza el Ratio de Sharpe.

    Parameters
    ----------
    weight : np.ndarray
        Matriz de pesos de los portafolios simulados.
    ReturnEsp : np.ndarray
        Rendimientos esperados.
    VolEsp : np.ndarray
        Volatilidades esperadas.
    RadioSharpe : np.ndarray
        Ratios de Sharpe.

    Returns
    -------
    best_weights : np.ndarray
        Pesos del portafolio óptimo.
    retorno : float
        Rendimiento esperado del mejor portafolio.
    volatilidad : float
        Volatilidad esperada del mejor portafolio.
    sharpe_ratio : float
        Ratio de Sharpe máximo.
    """
    max_index = int(RadioSharpe.argmax())
    best_weights = weight[max_index, :]
    return best_weights, float(ReturnEsp[max_index]), float(VolEsp[max_index]), float(RadioSharpe[max_index])


def encontrar_minima_varianza(
    weight: np.ndarray,
    ReturnEsp: np.ndarray,
    VolEsp: np.ndarray,
    RadioSharpe: np.ndarray,
) -> Tuple[np.ndarray, float, float, float]:
    """Identifica el portafolio con la mínima volatilidad (mínima varianza).

    Parameters
    ----------
    weight : np.ndarray
        Matriz de pesos de los portafolios simulados.
    ReturnEsp : np.ndarray
        Rendimientos esperados.
    VolEsp : np.ndarray
        Volatilidades esperadas.
    RadioSharpe : np.ndarray
        Ratios de Sharpe.

    Returns
    -------
    min_weights : np.ndarray
        Pesos del portafolio de mínima volatilidad.
    retorno : float
        Rendimiento esperado.
    volatilidad : float
        Volatilidad mínima alcanzada.
    sharpe_ratio : float
        Ratio de Sharpe asociado.
    """
    min_index = int(VolEsp.argmin())
    min_weights = weight[min_index, :]
    return min_weights, float(ReturnEsp[min_index]), float(VolEsp[min_index]), float(RadioSharpe[min_index])


def mostrar_resultados(
    tickers: List[str],
    best_weights: np.ndarray,
    retorno: float,
    volatilidad: float,
    sharpe_ratio: float,
) -> None:
    """Muestra en formato tabular o de lista los pesos y métricas del mejor portafolio."""
    print("=" * 50)
    print("MEJORES PESOS DEL PORTAFOLIO")
    print("=" * 50)
    for i, ticker in enumerate(tickers):
        print(f"  {ticker:<12}: {best_weights[i] * 100:>6.2f}%")
    print("-" * 50)
    print(f"  Retorno esperado   : {retorno:.6f}")
    print(f"  Volatilidad        : {volatilidad:.6f}")
    print(f"  Ratio Sharpe máximo: {sharpe_ratio:.4f}")
    print("=" * 50)


def desempeno(
    peso: np.ndarray,
    media_rendimiento: Union[pd.Series, np.ndarray],
    covmatrix: Union[pd.DataFrame, np.ndarray],
    time: int = 1,
) -> Tuple[float, float]:
    """Calcula el rendimiento acumulado esperado y la desviación estándar del portafolio.

    Parameters
    ----------
    peso : np.ndarray
        Vector de ponderaciones de los activos (suman 1).
    media_rendimiento : pd.Series or np.ndarray
        Media de rendimientos diarios de cada activo.
    covmatrix : pd.DataFrame or np.ndarray
        Matriz de covarianza de los rendimientos.
    time : int, default=1
        Horizonte de inversión temporal en días.

    Returns
    -------
    rendimiento : float
        Rendimiento esperado acumulado en el período `time`.
    std : float
        Desviación estándar (riesgo) acumulada en el período `time`.
    """
    w = np.asarray(peso)
    mu = np.asarray(media_rendimiento)
    sigma = np.asarray(covmatrix)

    rendimiento = float(np.sum(mu * w) * time)
    std = float(np.sqrt(np.dot(w.T, np.dot(sigma, w))) * np.sqrt(time))
    return rendimiento, std


# Alias amigable
portfolio_performance = desempeno


def ejecutar_analisis(
    tickers: List[str],
    start_date: str,
    end_date: str,
    num_portafolios: int = 5000,
) -> Tuple[np.ndarray, float, float, float]:
    """Flujo completo: descarga datos, calcula rendimientos, simula portafolios
    y muestra los resultados óptimos.
    """
    df_aj = descargar_datos(tickers, start_date, end_date)
    log_returns = calcular_rendimientos_log(df_aj)
    weight, ReturnEsp, VolEsp, RadioSharpe = simular_portafolios(log_returns, num_portafolios=num_portafolios)
    best_weights, retorno, volatilidad, sharpe_ratio = encontrar_mejor_portafolio(
        weight, ReturnEsp, VolEsp, RadioSharpe
    )
    mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio)
    return best_weights, retorno, volatilidad, sharpe_ratio
