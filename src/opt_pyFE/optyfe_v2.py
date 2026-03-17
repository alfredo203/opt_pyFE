# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 13:44:27 2025

@author: david
"""

#importamos el paquete de optimización de portafolios
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import glob


#Definimos los parametros con los que trabajaremos y las especificaciones
#de nuestro analisis 

# Lista de los tickers que analizaremos para nuestro portafolio 
tickers = ['GOOGL',
'NVDA',
'PG',
'ETSY',
'CVS',
'PARA',
'VZ',
'SITES1A-1.MX',
'CCL',
'KIMBERA.MX',
'BABA',
'MEGACPO.MX',
'GM',
'AMAT',
'MARA'] 
 
# Fecha de inicio del análisis
start_date = "2023-05-26"  
# Fecha final del análisis
end_date = "2024-05-26"   

num_portafolios=5000

time = 100
inversion_inicial = 10000
mc_sims = 400 # numero de simulaciones

path = 'C:/Users/david/Documents/Programación/datos_optpy'


def getdata(path, tickers, start_date, end_date): 
    stockdata = glob.glob(path + '/*.csv')
    df = pd.concat((pd.read_csv(stock) for stock in stockdata), ignore_index=True)

    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df_aj = df[(df["Fecha"] >= start_date) & (df["Fecha"] <= end_date)]
    
    # Filtrar por tickers si existe columna "Ticker"
    if 'Ticker' in df_aj.columns:
        df_aj = df_aj[df_aj['Ticker'].isin(tickers)]
    
    # Formato pivot (ancho, 1 columna por ticker)
    df_wide = df_aj.pivot(index='Fecha', columns='Ticker', values='Cierre')
    
    # Formato largo (Fecha, Ticker, Cierre) -> lo que proyeccion espera
    df_long = df_aj[['Fecha', 'Ticker', 'Cierre']].reset_index(drop=True)
    
    return df_wide, df_long


df_wide, df_long = getdata(path, tickers, start_date, end_date)


def proyeccion(df_long, window=50):
    pendientes = {}

    # Recorremos cada ticker único
    for ticker in df_long['Ticker'].unique():
        # Filtrar datos del ticker actual
        serie = df_long[df_long['Ticker'] == ticker].copy()
        serie = serie.sort_values('Fecha').reset_index(drop=True)

        # Crear columna de días
        serie['Día'] = np.arange(1, len(serie) + 1)

        # Calcular media móvil
        serie['MA_50'] = serie['Cierre'].rolling(window=window).mean()

        # Variables independientes y dependientes
        X = serie[['Día']]
        y = serie['Cierre']

        # Crear modelo de regresión
        regresion = LinearRegression()
        regresion.fit(X, y)

        # Guardar pendiente
        pendiente = regresion.coef_[0]
        pendientes[ticker] = pendiente
        print(f"Pendiente de {ticker}: {pendiente:.4f}")

        # Predicciones
        y_pred = regresion.predict(X)

        # Gráfico
        plt.figure(figsize=(10, 6))
        plt.plot(serie['Día'], serie['Cierre'], color='blue', label='Precio de Cierre')
        plt.plot(serie['Día'], serie['MA_50'], color='green', label='Media Móvil 50 días')
        plt.plot(serie['Día'], y_pred, color='red', label='Línea de Regresión')
        plt.title(f'Regresión Lineal y Media Móvil para {ticker}')
        plt.xlabel('Día')
        plt.ylabel('Precio de Cierre')
        plt.legend()
        plt.grid(True)
        plt.show()

    return pendientes

pendientes = proyeccion(df_long)


# Bandas de Bollinger

def bandas_bollinger(df_long, window=20):    
    # Se crea una variables de los datos
    datos = {}

    for ticker in df_long['Ticker'].unique():
        # Filtrar datos del ticker actual
        data = df_long[df_long['Ticker'] == ticker].copy()
        data = data.sort_values('Fecha').reset_index(drop=True)
        
        # Calculo de la media movil y desviacion estandar
        data['MA20'] = data['Cierre'].rolling(window=window).mean()
        data['std_dev'] = data['Cierre'].rolling(window=window).std()
        
        # Calculo de las bandas de Bollinger
        data['UpperBand'] = data['MA20'] + 2 * data['std_dev']
        data['MiddleBand'] = data['MA20']
        data['LowerBand'] = data['MA20'] - 2 * data['std_dev']
        
        # Guardar el DataFrame en el diccionario
        datos[ticker] = data
        
        # Grafico individual para cada ticker
        plt.figure(figsize=(10, 6))
        plt.plot(data['Cierre'], label='Precio de cierre', linewidth=1.5)
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

datos = bandas_bollinger(df_long)

# Ejemplo de uso
# tickers = ['AAPL', 'MSFT']  # Puedes agregar más tickers
# start_date = '2024-01-01'
# end_date = '2024-06-01'

# bandas_bollinger(tickers, start_date, end_date)


# # Rendimientos
    # rendimiento = df_aj.pct_change()
    # media_rendimiento = rendimiento.mean()
    # covmatrix = rendimiento.cov()

    # rendimiento, media_rendimiento, covmatrix
    
def markowitz(df_wide, num_portafolios = 5000):
    # Calcula los rendimientos simples porcentuales   
    rendimiento = df_wide.pct_change()
    # Convierte los rendimientos simples en rendimientos logarítmicos
    log_returns = np.log(1 + rendimiento)
    # Retorna los rendimientos logarítmicos
   
    num_activos = log_returns.shape[1]
    # Inicializa matrices para almacenar los pesos, rendimientos, volatilidades y ratios Sharpe
    weight = np.zeros((num_portafolios, num_activos))
    ReturnEsp = np.zeros(num_portafolios)
    VolEsp = np.zeros(num_portafolios)
    RadioSharpe = np.zeros(num_portafolios)

    # Calcula la media y covarianza de los rendimientos logarítmicos
    meanlogReturns = log_returns.mean()
    Sigma = log_returns.cov()

    # Simulación de los portafolios
    for k in range(num_portafolios):
        # Genera pesos aleatorios para los activos y los normaliza a 1
        w = np.random.random(num_activos)
        w /= np.sum(w)
        # Almacena los pesos en la matriz correspondiente
        weight[k, :] = w

        # Calcula el rendimiento esperado del portafolio con los pesos aleatorios
        ReturnEsp[k] = np.sum(meanlogReturns * w)
        # Calcula la volatilidad esperada del portafolio
        VolEsp[k] = np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
        # Calcula el Ratio Sharpe del portafolio
        RadioSharpe[k] = ReturnEsp[k] / VolEsp[k]

    # --- Identificar el mejor portafolio ---
    max_index = RadioSharpe.argmax()
    
    best_weights = weight[max_index, :]
    best_return = ReturnEsp[max_index]
    best_volatility = VolEsp[max_index]
    best_sharpe = RadioSharpe[max_index]
# Función para mostrar los resultados del mejor portafolio
    # Muestra los pesos de los activos en el mejor portafolio
    print("Mejores pesos del portafolio:")
    for i, col in enumerate(df_wide.columns):
        print(f"{col}: {best_weights[i] * 100:.2f}%")
        # Muestra el rendimiento esperado, la volatilidad y el Ratio Sharpe del mejor portafolio
    print(f"Retorno esperado del portafolio: {best_return:.4f}")
    print(f"Volatilidad esperada del portafolio: {best_volatility:.4f}")
    print(f"Ratio Sharpe máximo: {best_sharpe:.4f}")
    
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(VolEsp, ReturnEsp, c=RadioSharpe, cmap="viridis", s=10, alpha=0.5)
    plt.colorbar(sc, label="Sharpe Ratio")
    plt.scatter(best_volatility, best_return, c="red", marker="*", s=200, label="Mejor Sharpe")
    plt.xlabel("Volatilidad esperada (σ)")
    plt.ylabel("Retorno esperado (μ)")
    plt.title("Simulación Montecarlo de Portafolios y Frontera Eficiente")
    plt.legend()
    plt.show()
    
    return best_weights

pesos = markowitz(df_wide)



def hVaR(df_wide, peso, time, inversion_inicial, alpha = 5 ):
    
    peso /= np.sum(peso) #redondeamos los pesos para que sumen 1
    rendimiento = df_wide.pct_change().dropna()
    media_rendimiento = rendimiento.mean()

    rendimiento_esp = np.sum(media_rendimiento*peso)*time 
    rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 

    VaR = np.percentile(rendimiento["portafolio"], alpha)
    hVaR = round(-VaR * np.sqrt(time) * inversion_inicial, 2)

    belowVar = rendimiento["portafolio"] <= VaR
    CVaR = rendimiento["portafolio"][belowVar].mean()
    hCVaR = round(-CVaR * np.sqrt(time) * inversion_inicial, 2)
    
    print("\nVaR:")

    print(' historical VaR 95th CI   :    ', round(hVaR))
    print(' historical CVaR 95th CI  :    ', round(hCVaR))
    
    print("\nPortfolio:")

    print(' initial portfolio         :    ', inversion_inicial)
    print(' portfolio performance     :    ', inversion_inicial*rendimiento_esp)

def McVaR(df_wide, peso, time, initialPortfolio, mc_sims, alpha = 5):
    
    rendimiento = df_wide.pct_change().dropna()
    media_rendimiento = rendimiento.mean()
    covmatrix = rendimiento.cov()
    # Verificar que media_rendimiento sea un vector
    meanM = np.tile(media_rendimiento, (time, 1))  # T filas y una copia de media_rendimiento en cada fila
    portfolio_sims = np.zeros((time, mc_sims))  # Inicializa la matriz para guardar simulaciones

    for m in range(mc_sims):
        # Generar rendimientos diarios simulados
        Z = np.random.normal(size=(time, len(peso)))
        L = np.linalg.cholesky(covmatrix)
        dailyReturns = meanM + Z @ L.T  # Multiplicación de matrices para simular rendimientos correlacionados

        # Cálculo de la evolución del portafolio
        portfolio_returns = np.cumprod(1 + np.dot(dailyReturns, peso))  # Acumular rendimientos diarios
        portfolio_sims[:, m] = portfolio_returns * initialPortfolio  # Aplicar valor inicial del portafolio
        
    plt.plot(portfolio_sims)
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Days')
    plt.title('MC simulation of a stock portfolio')
    plt.show()
        
    portResults = pd.Series(portfolio_sims[-1,:])
    
    VaR = np.percentile(portResults, alpha)
    MCVaR = inversion_inicial - VaR
    print("MC VaR 95th CI          :    ", round(MCVaR, 2))
    
    belowVaR = portResults <= VaR
    CVaR = portResults[belowVaR].mean()
    MCCVaR = inversion_inicial - CVaR
    print("MC CVaR 95th CI          :    ", round(MCCVaR, 2))


hVaR(df_wide, pesos, time, inversion_inicial)
McVaR(df_wide, pesos, time, inversion_inicial, mc_sims)

