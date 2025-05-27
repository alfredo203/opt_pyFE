# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:19:16 2025

@author: David
"""

import yfinance as yf
import matplotlib.pyplot as plt

def bandas_bollinger(tickers, start_date, end_date, window=20):
    
    # pone las acciones en formato de lista para poder precesar mas de uno
    if isinstance(tickers, str):  
        tickers = [tickers]  
    
    # Se crea una variables de los datos
    datos = {}

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Calculo de la media movil y desviacion estandar
        data['MA20'] = data['Close'].rolling(window=window).mean()
        data['std_dev'] = data['Close'].rolling(window=window).std()
        
        # Calculo de las bandas de Bollinger
        data['UpperBand'] = data['MA20'] + 2 * data['std_dev']
        data['MiddleBand'] = data['MA20']
        data['LowerBand'] = data['MA20'] - 2 * data['std_dev']
        
        # Guardar el DataFrame en el diccionario
        datos[ticker] = data
        
        # Grafico individual para cada ticker
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Precio de cierre', linewidth=1.5)
        plt.plot(data['UpperBand'], 
                 linestyle='--', 
                 linewidth=1, 
                 color = "red",
                 label='Banda Superior')
        plt.plot(data['MiddleBand'], 
                 linestyle='--', 
                 linewidth=1,
                 color = "blue",
                 label='Banda Media')
        plt.plot(data['LowerBand'], 
                 linestyle='--', 
                 linewidth=1, 
                 color = "red",
                 label='Banda Inferior')

        plt.title(f'Bandas de Bollinger - {ticker}')
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.legend()
        plt.show()
    
    return datos  # Devuelve un diccionario con los DataFrames de cada ticker

# Ejemplo de uso
tickers = ['AAPL', 'MSFT']  # Puedes agregar más tickers
start_date = '2024-01-01'
end_date = '2024-06-01'

bandas_bollinger(tickers, start_date, end_date)
