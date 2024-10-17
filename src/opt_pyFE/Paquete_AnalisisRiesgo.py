# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:43:30 2024

@author: david
"""

import pandas as pd #manejo de dataframes
import numpy as np #operaciones matematicas y generacion de num aleatorios
import matplotlib.pyplot as plt #creacion de graficos 
import yfinance as yf #obtencion de datos financieros 
import datetime as dt #trabajar con fechas


#getdata exatre los valores de cierre de las emisoras espificadas en tickers
#devuelve el cambio porcentual de la serie (rendimiento), la media del 
#cambio y la matriz de covarianza de las emisoras 
def getdata(stocks, start, end): 
    stockdata =yf.download(stocks, start=start, end=end) 
    stockdata = stockdata['Close'] 
    rendimiento = stockdata.pct_change() 
    media_rendimiento = rendimiento.mean()
    covmatrix = rendimiento.cov()
    return rendimiento, media_rendimiento, covmatrix 

#desempeno calcula el rendimiento de nuestro portafolio y la desviacion 
#estandar del mismo 
def desempeno(peso, media_rendimiento, covmatrix, time):
    rendimiento = np.sum(media_rendimiento*peso)*time 
    std = np.sqrt(np.dot(peso.T, np.dot(covmatrix, peso))) * np.sqrt(time)
    return rendimiento, std 

def historicalVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        historicalVar = np.percentile(rendimiento,alpha)
    elif isinstance(rendimiento, pd.DataFrame):
        historicalVar = rendimiento.aggregate(historicalVar, alpha=alpha)
    else :
        raise TypeError('Se espera que rendimiento sea dataframe o serie')
        
    hVaR = -historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
    return hVaR
        
        
#calcula el valor VaR historico condicional (los valores que exceden la medida
#del VaR historico) dado un intervalo de confianza alpha
def historicalCVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVar = rendimiento <= historicalVar(rendimiento, alpha=alpha)
        historicalCVar = rendimiento[belowVar].mean()
    elif isinstance(rendimiento, pd.DataFrame):
        historicalCVar = rendimiento.aggregate(historicalCVar, alpha=alpha)
    else:
        raise TypeError('Se espera que rendimiento sea dataframe o serie')
        
    hCVaR = -historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
    return hCVaR
        

def MonteCarlo(mc_sims, T, media_rendimiento, peso, initialPortfolio, covmatrix):
    # Verificar que media_rendimiento sea un vector
    meanM = np.tile(media_rendimiento, (T, 1))  # T filas y una copia de media_rendimiento en cada fila
    portfolio_sims = np.zeros((T, mc_sims))  # Inicializa la matriz para guardar simulaciones

    for m in range(mc_sims):
        # Generar rendimientos diarios simulados
        Z = np.random.normal(size=(T, len(peso)))
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

    return portfolio_sims, portResults

def montecVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        VaR = np.percentile(rendimiento, alpha)
    else:
        raise TypeError("Se espera una serie de datos de pandas")
        
    mcVaR = initialPortfolio - VaR(portResults, alpha=5)
    return mcVaR

def montecCVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVaR = rendimiento <= mcVaR(rendimiento, alpha=alpha)
        CVaR = rendimiento[belowVaR].mean()
    else:
        raise TypeError("Se espera una serie de datos de pandas")
        
    mcCVaR = initialPortfolio - CVaR(portResults, alpha=5)
    return mcCVaR
               
def resum() :
    print("\nVaR:")

    print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))
    print(" MC VaR  95th CI          :    ", round(mcVaR, 2))

    print("\nCVaR:")

    print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))
    print(" MC CVaR 95th CI          :    ", round(mcCVaR, 2))

    print("\Portfolio")

    print('initial portfolio         :    ', round(initialPortfolio))
    print('portfolio performance     :    ', round(pRet*inversion_inicial))