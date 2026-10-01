"""Módulo para la descarga y preparación de datos financieros."""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yfinance as yf


def descargar_datos(
    tickers: Union[str, List[str]],
    start_date: str,
    end_date: str,
    datos_req: Optional[Union[str, List[str]]] = "Close",
) -> pd.DataFrame:
    """Descarga datos históricos de precios desde Yahoo Finance.

    Parameters
    ----------
    tickers : str or list of str
        Ticker o lista de tickers de activos financieros (ej. 'AAPL' o ['AAPL', 'MSFT']).
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'.
    end_date : str
        Fecha final en formato 'YYYY-MM-DD'.
    datos_req : str or list of str, optional
        Columna(s) solicitada(s) (ej. 'Close', 'Adj Close'). Si es None, retorna todo el DataFrame.
        Por defecto es 'Close'.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos descargados e indexados por fecha.
    """
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = list(tickers)

    df = yf.download(
        tickers_list,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    if datos_req is not None:
        if isinstance(datos_req, str):
            col_target = datos_req
            if col_target in df.columns:
                return df[col_target]
            elif isinstance(df.columns, pd.MultiIndex):
                # Caso multi-index (ej. nivel 0 = 'Close', nivel 1 = Tickers)
                if col_target in df.columns.levels[0]:
                    return df[col_target]
        else:
            # Lista de columnas
            cols_present = [c for c in datos_req if c in df.columns]
            if cols_present:
                return df[cols_present]

    return df


def getdata(
    stocks: Union[str, List[str]],
    start: Union[str, object],
    end: Union[str, object],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Descarga los precios de cierre y calcula rendimientos simples porcentuales,
    su media y la matriz de covarianza.

    Parameters
    ----------
    stocks : str or list of str
        Lista de tickers o un solo ticker.
    start : str or datetime
        Fecha inicial.
    end : str or datetime
        Fecha final.

    Returns
    -------
    rendimiento : pd.DataFrame
        Rendimientos porcentuales diarios (pct_change).
    media_rendimiento : pd.Series
        Media aritmética de rendimientos diarios por activo.
    covmatrix : pd.DataFrame
        Matriz de covarianza de los rendimientos diarios.
    """
    if isinstance(stocks, str):
        stocks = [stocks]

    stockdata = yf.download(stocks, start=start, end=end, progress=False, auto_adjust=False)

    if stockdata.empty:
        empty_df = pd.DataFrame()
        return empty_df, pd.Series(dtype=float), empty_df

    if isinstance(stockdata.columns, pd.MultiIndex):
        if "Close" in stockdata.columns.levels[0]:
            stockdata = stockdata["Close"]
    elif "Close" in stockdata.columns:
        stockdata = stockdata["Close"]

    rendimiento = stockdata.pct_change().dropna()
    media_rendimiento = rendimiento.mean()
    covmatrix = rendimiento.cov()

    return rendimiento, media_rendimiento, covmatrix


# Alias amigable con PEP 8
get_data = getdata


def calcular_rendimientos_log(df_precios: pd.DataFrame) -> pd.DataFrame:
    """Calcula rendimientos logarítmicos continuos a partir de un DataFrame de precios.

    Parameters
    ----------
    df_precios : pd.DataFrame
        DataFrame de precios con índice temporal.

    Returns
    -------
    pd.DataFrame
        Rendimientos logarítmicos: ln(P_t / P_{t-1}).
    """
    # Usar razón directa evita RuntimeWarning por valores negativos o división por 0
    shifted = df_precios.shift(1)
    ratio = df_precios / shifted
    log_returns = np.log(ratio)
    return log_returns.dropna()
