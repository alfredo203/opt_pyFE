"""Módulo para la medición y gestión de riesgos financieros (VaR, CVaR y Monte Carlo)."""

from typing import Dict, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def historical_var(
    rendimiento: Union[pd.Series, pd.DataFrame],
    alpha: float = 5.0,
) -> Union[float, pd.Series]:
    """Calcula el Valor en Riesgo (VaR) histórico dado un nivel de significancia alpha.

    Parameters
    ----------
    rendimiento : pd.Series or pd.DataFrame
        Serie o DataFrame de rendimientos históricos.
    alpha : float, default=5.0
        Nivel de significancia porcentual (ej. 5 para un nivel de confianza del 95%).

    Returns
    -------
    float or pd.Series
        Cuantil que representa el VaR histórico.
    """
    if isinstance(rendimiento, pd.Series):
        return float(np.percentile(rendimiento.dropna(), alpha))
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.dropna().apply(lambda s: float(np.percentile(s, alpha)))
    else:
        raise TypeError("Se espera que rendimiento sea un pandas Series o DataFrame.")


# Alias retrocompatible
historicalVar = historical_var


def historical_cvar(
    rendimiento: Union[pd.Series, pd.DataFrame],
    alpha: float = 5.0,
) -> Union[float, pd.Series]:
    """Calcula el Valor en Riesgo Condicional (CVaR / Expected Shortfall) histórico.

    Parameters
    ----------
    rendimiento : pd.Series or pd.DataFrame
        Serie o DataFrame de rendimientos históricos.
    alpha : float, default=5.0
        Nivel de significancia porcentual (ej. 5 para un nivel de confianza del 95%).

    Returns
    -------
    float or pd.Series
        Promedio de los rendimientos que se encuentran por debajo o iguales al VaR.
    """
    if isinstance(rendimiento, pd.Series):
        s = rendimiento.dropna()
        var_threshold = np.percentile(s, alpha)
        tail = s[s <= var_threshold]
        return float(tail.mean()) if not tail.empty else float(var_threshold)
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.dropna().apply(
            lambda s: float(s[s <= np.percentile(s, alpha)].mean())
            if not s[s <= np.percentile(s, alpha)].empty
            else float(np.percentile(s, alpha))
        )
    else:
        raise TypeError("Se espera que rendimiento sea un pandas Series o DataFrame.")


# Alias retrocompatible
historicalCVar = historical_cvar


def monte_carlo_sim(
    mc_sims: int,
    T: int,
    media_rendimiento: Union[pd.Series, np.ndarray],
    peso: Union[pd.Series, np.ndarray],
    initialPortfolio: float,
    covmatrix: Union[pd.DataFrame, np.ndarray],
    plot: bool = False,
) -> Tuple[np.ndarray, pd.Series]:
    """Ejecuta una simulación Monte Carlo multivariada para la evolución del valor del portafolio.

    Parameters
    ----------
    mc_sims : int
        Número de trayectorias a simular.
    T : int
        Horizonte de tiempo en días.
    media_rendimiento : pd.Series or np.ndarray
        Vector de rendimientos diarios esperados de cada activo.
    peso : pd.Series or np.ndarray
        Vector de pesos del portafolio.
    initialPortfolio : float
        Monto o valor monetario inicial del portafolio.
    covmatrix : pd.DataFrame or np.ndarray
        Matriz de covarianza de los rendimientos.
    plot : bool, default=False
        Si es True, grafica las trayectorias simuladas.

    Returns
    -------
    portfolio_sims : np.ndarray
        Matriz de dimensiones (T, mc_sims) con los valores simulados del portafolio día a día.
    portResults : pd.Series
        Serie con los valores finales del portafolio en el día T para cada simulación.
    """
    mu = np.asarray(media_rendimiento)
    w = np.asarray(peso)
    cov = np.asarray(covmatrix)

    n_activos = len(w)
    meanM = np.tile(mu, (T, 1))
    portfolio_sims = np.zeros((T, mc_sims))

    # Cholesky con fallback a regularización espectral si no es estrictamente positiva definida
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Regularización Ridge mínima
        cov_reg = cov + np.eye(n_activos) * 1e-7
        try:
            L = np.linalg.cholesky(cov_reg)
        except np.linalg.LinAlgError:
            # Descomposición en valores propios garantizando semidefinición positiva
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    for m in range(mc_sims):
        Z = np.random.normal(size=(T, n_activos))
        dailyReturns = meanM + Z @ L.T
        portfolio_returns = np.cumprod(1 + np.dot(dailyReturns, w))
        portfolio_sims[:, m] = portfolio_returns * initialPortfolio

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_sims)
        plt.ylabel("Portfolio Value ($)")
        plt.xlabel("Days")
        plt.title("MC simulation of a stock portfolio")
        plt.grid(True)
        plt.show()

    portResults = pd.Series(portfolio_sims[-1, :])
    return portfolio_sims, portResults


# Alias retrocompatible
MonteCarlo = monte_carlo_sim


def mc_var(
    portResults: pd.Series,
    alpha: float = 5.0,
) -> float:
    """Calcula el cuantil percentil en la simulación Monte Carlo para el valor en riesgo."""
    if not isinstance(portResults, pd.Series):
        portResults = pd.Series(portResults)
    return float(np.percentile(portResults.dropna(), alpha))


# Alias retrocompatible
mcVaR = mc_var


def mc_cvar(
    portResults: pd.Series,
    alpha: float = 5.0,
) -> float:
    """Calcula el valor en riesgo condicional (CVaR) en la simulación Monte Carlo."""
    if not isinstance(portResults, pd.Series):
        portResults = pd.Series(portResults)
    s = portResults.dropna()
    var_val = mc_var(s, alpha=alpha)
    below_var = s[s <= var_val]
    return float(below_var.mean()) if not below_var.empty else var_val


# Alias retrocompatible
mcCVaR = mc_cvar


def resumen_riesgo(
    inversion_inicial: float,
    initialPortfolio: float,
    hVaR: float,
    hCVaR: float,
    MCVaR: float,
    MCCVaR: float,
    pRet: float,
) -> Dict[str, float]:
    """Imprime y retorna un diccionario con el resumen estructurado de las métricas de riesgo."""
    res = {
        "historical_VaR_monto": round(inversion_inicial * hVaR, 2),
        "MC_VaR_monto": round(MCVaR, 2),
        "historical_CVaR_monto": round(inversion_inicial * hCVaR, 2),
        "MC_CVaR_monto": round(MCCVaR, 2),
        "initial_portfolio": round(initialPortfolio, 2),
        "portfolio_performance_monto": round(pRet * inversion_inicial, 2),
    }

    print("\n" + "=" * 45)
    print("           RESUMEN DE RIESGO")
    print("=" * 45)
    print("  VaR (95% CI):")
    print(f"    - Historical VaR  : ${res['historical_VaR_monto']:>12,.2f}")
    print(f"    - Monte Carlo VaR : ${res['MC_VaR_monto']:>12,.2f}")
    print("\n  CVaR (95% CI):")
    print(f"    - Historical CVaR : ${res['historical_CVaR_monto']:>12,.2f}")
    print(f"    - Monte Carlo CVaR: ${res['MC_CVaR_monto']:>12,.2f}")
    print("\n  Portafolio:")
    print(f"    - Valor Inicial   : ${res['initial_portfolio']:>12,.2f}")
    print(f"    - Rendimiento Est.: ${res['portfolio_performance_monto']:>12,.2f}")
    print("=" * 45)

    return res


# Alias retrocompatible
resum = resumen_riesgo
