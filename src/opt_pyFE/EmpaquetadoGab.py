# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:54:19 2024

@author: gayal
"""
# Importar librerías necesarias
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Definir el rango de fechas, en este caso es un programa para medias móviles
#a 20 días
start_date = '2024-01-01'
end_date = '2024-09-23'

# Primera versión de proyección: 
# def proyección(ticker):
#     # Descargar datos del ticker actual
#     data = yf.download(ticker, start=start_date, end=end_date)
    
#     # Comprobar si hay datos suficientes
#     if not data.empty:
#         # Crear DataFrame y agregar columna de días
#         df = pd.DataFrame(data)
#         df['Día'] = np.arange(1, len(df) + 1)
        
#         # Calcular la media móvil de los últimos 50 días
#         df['MA_50'] = df['Close'].rolling(window=20).mean()

#         # Imprimir la tabla con las columnas relevantes
#         print(f"\nTabla de Precios y Media Movil de 50 días para {ticker}:\n")
#         print(df[['Día', 'Close', 'MA_50']].tail(40))  # Mostrar las últimas 40 filas
#     else:
#         print(f"No se encontraron datos para {ticker}")

# Sobrescribir clean_data: ejecutar regresión y generar gráficos
def proyeccion(tickers, window = 50):
    # Descargar datos del tickers actual
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Comprobar si hay datos suficientes
    if not data.empty:
        # Crear DataFrame y agregar columna de días
        df = pd.DataFrame(data)
        df['Día'] = np.arange(1, len(df) + 1)
        
        # Calcular la media móvil de los últimos 50 días
        df['MA_50'] = df['Close'].rolling(window=window).mean()

        # Variables independientes (Día) y dependientes (Precio de Close)
        X = df[['Día']]  # Días como variable independiente
        y = df['Close']  # Precio de cierre como variable dependiente
        
        # Crear el modelo de regresión lineal
        regresion = LinearRegression()
        
        # Entrenar el modelo
        regresion.fit(X, y)
        
        # Obtener la pendiente (coeficiente) y agregarla a la lista
        pendientes = []
        pendiente = regresion.coef_[0]
        pendientes.append((tickers, pendiente))
        print(f"Pendiente de {tickers}: {pendiente}")
        
        # Hacer predicciones para los días de prueba
        y_pred = regresion.predict(X)
        
        # Generar gráfica
        plt.figure(figsize=(10, 6))
        
        # Gráfico de precios de cierre
        plt.plot(df['Día'], df['Close'], color='blue', label='Precio de Cierre')
        
        # Gráfico de la media móvil de 50 días
        plt.plot(df['Día'], df['MA_50'], color='green', label='Media Móvil 50 días')
        
        # Gráfico de la regresión lineal
        plt.plot(df['Día'], y_pred, color='red', label='Línea de Regresión')
        
        plt.title(f'Regresión Lineal y Media Móvil para {tickers} (Precio de Cierre)')
        plt.xlabel('Día')
        plt.ylabel('Precio de Cierre')
        plt.legend()
        plt.grid(True)
        
        # Mostrar la gráfica
        plt.show()
    else:
        print(f"No se encontraron datos para {tickers}")    
    return pendientes

# Ejemplo con 2 tickers

# tickers = ['NVDA', 'AAPL']

# # Bucle para procesar cada ticker
# for ticker in tickers:
#     proyeccion(ticker)

# OBSOLETO NO USAR
# Mostrar todas las pendientes almacenadas
# print("\nPendientes de las acciones:")
# for ticker, pendiente in pendientes:
#     print(f"{ticker}: {pendiente}")
