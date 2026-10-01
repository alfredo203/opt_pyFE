"""Módulo de analítica técnica: medias móviles, regresión lineal y bandas de Bollinger."""

from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import yfinance as yf


def proyeccion(
    tickers: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    window: int = 50,
    plot: bool = True,
    data: Optional[pd.DataFrame] = None,
) -> List[Tuple[str, float]]:
    """Calcula la media móvil y ajusta una regresión lineal sobre los precios de cierre.

    Parameters
    ----------
    tickers : str or list of str
        Ticker(s) a procesar (ej. 'AAPL' o ['AAPL', 'MSFT']).
    start_date : str, optional
        Fecha inicial ('YYYY-MM-DD'). Requerida si no se provee `data`.
    end_date : str, optional
        Fecha final ('YYYY-MM-DD'). Requerida si no se provee `data`.
    window : int, default=50
        Ventana de días para el cálculo de la media móvil.
    plot : bool, default=True
        Si es True, genera y muestra la gráfica comparativa.
    data : pd.DataFrame, optional
        DataFrame pre-descargado para evitar llamadas a la red.

    Returns
    -------
    list of tuple
        Lista de tuplas `(ticker, pendiente)` obtenidas de la regresión lineal.
    """
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    pendientes: List[Tuple[str, float]] = []

    for ticker in tickers_list:
        if data is not None:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data["Close"].columns:
                    ticker_series = data["Close"][ticker]
                else:
                    continue
            elif "Close" in data.columns:
                ticker_series = data["Close"]
            elif ticker in data.columns:
                ticker_series = data[ticker]
            else:
                ticker_series = data.iloc[:, 0]
            df_aj = pd.DataFrame({"Close": ticker_series.dropna()})
        else:
            if not start_date or not end_date:
                raise ValueError("Debe proveer 'start_date' y 'end_date' si no proporciona 'data'.")
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if raw_data.empty:
                print(f"No se encontraron datos para {ticker}")
                continue
            if isinstance(raw_data.columns, pd.MultiIndex):
                df_aj = pd.DataFrame({"Close": raw_data["Close"][ticker]})
            elif "Close" in raw_data.columns:
                df_aj = pd.DataFrame({"Close": raw_data["Close"]})
            else:
                df_aj = pd.DataFrame({"Close": raw_data.iloc[:, 0]})

        if df_aj.empty or len(df_aj) < 2:
            print(f"Datos insuficientes para {ticker}")
            continue

        df_aj = df_aj.copy()
        df_aj["Día"] = np.arange(1, len(df_aj) + 1)
        df_aj[f"MA_{window}"] = df_aj["Close"].rolling(window=window).mean()

        X = df_aj[["Día"]]
        y = df_aj["Close"]

        regresion = LinearRegression()
        regresion.fit(X, y)
        pendiente = float(regresion.coef_[0])
        pendientes.append((ticker, pendiente))

        y_pred = regresion.predict(X)

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df_aj["Día"], df_aj["Close"], color="blue", label="Precio de Cierre")
            plt.plot(df_aj["Día"], df_aj[f"MA_{window}"], color="green", label=f"Media Móvil {window} días")
            plt.plot(df_aj["Día"], y_pred, color="red", label="Línea de Regresión")
            plt.title(f"Regresión Lineal y Media Móvil para {ticker} (Precio de Cierre)")
            plt.xlabel("Día")
            plt.ylabel("Precio de Cierre")
            plt.legend()
            plt.grid(True)
            plt.show()

    return pendientes


def bandas_bollinger(
    tickers: Union[str, List[str]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    window: int = 20,
    num_std: float = 2.0,
    plot: bool = True,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Calcula y grafica las bandas de Bollinger para uno o más activos.

    Parameters
    ----------
    tickers : str or list of str
        Ticker o lista de tickers.
    start_date : str, optional
        Fecha inicial ('YYYY-MM-DD').
    end_date : str, optional
        Fecha final ('YYYY-MM-DD').
    window : int, default=20
        Ventana de días para el cálculo de la media móvil y desviación estándar.
    num_std : float, default=2.0
        Número de desviaciones estándar para las bandas superior e inferior.
    plot : bool, default=True
        Si es True, genera la gráfica de bandas de Bollinger para cada ticker.
    data : pd.DataFrame, optional
        DataFrame pre-cargado de precios.

    Returns
    -------
    dict of {str: pd.DataFrame}
        Diccionario donde cada clave es un ticker y su valor es el DataFrame con
        las columnas: 'Close', 'MA20', 'std_dev', 'UpperBand', 'MiddleBand', 'LowerBand'.
    """
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    datos: Dict[str, pd.DataFrame] = {}

    for ticker in tickers_list:
        if data is not None:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker in data["Close"].columns:
                    ticker_series = data["Close"][ticker]
                else:
                    continue
            elif ticker in data.columns:
                ticker_series = data[ticker]
            elif "Close" in data.columns:
                ticker_series = data["Close"]
            else:
                ticker_series = data.iloc[:, 0]
            df = pd.DataFrame({"Close": ticker_series.dropna()})
        else:
            if not start_date or not end_date:
                raise ValueError("Debe proveer 'start_date' y 'end_date' si no proporciona 'data'.")
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if raw_data.empty:
                print(f"No se encontraron datos para {ticker}")
                continue
            if isinstance(raw_data.columns, pd.MultiIndex):
                df = pd.DataFrame({"Close": raw_data["Close"][ticker]})
            elif "Close" in raw_data.columns:
                df = pd.DataFrame({"Close": raw_data["Close"]})
            else:
                df = pd.DataFrame({"Close": raw_data.iloc[:, 0]})

        df = df.copy()
        ma_col = f"MA{window}"
        df[ma_col] = df["Close"].rolling(window=window).mean()
        df["std_dev"] = df["Close"].rolling(window=window).std()
        df["UpperBand"] = df[ma_col] + num_std * df["std_dev"]
        df["MiddleBand"] = df[ma_col]
        df["LowerBand"] = df[ma_col] - num_std * df["std_dev"]

        datos[ticker] = df

        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df["Close"], label="Precio de cierre", linewidth=1.5)
            plt.plot(df.index, df["UpperBand"], linestyle="--", linewidth=1, color="red", label="Banda Superior")
            plt.plot(df.index, df["MiddleBand"], linestyle="--", linewidth=1, color="blue", label="Banda Media")
            plt.plot(df.index, df["LowerBand"], linestyle="--", linewidth=1, color="red", label="Banda Inferior")
            plt.title(f"Bandas de Bollinger - {ticker}")
            plt.xlabel("Fecha")
            plt.ylabel("Precio")
            plt.legend()
            plt.grid(True)
            plt.show()

    return datos


# Alias amigable
bollinger_bands = bandas_bollinger
