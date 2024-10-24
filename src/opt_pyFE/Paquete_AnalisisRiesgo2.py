# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:12:20 2024

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

#tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #creamos un portafolio

#end_date = dt.datetime.now()
#start_date = end_date - dt.timedelta(days=800)

#rendimiento, rendimiento_medio, covmatrix = getdata(tickers, start_date, end_date)
#rendimiento = rendimiento.dropna() #eliminamos los valores nulos 

#desempeno calcula el rendimiento de nuestro portafolio y la desviacion 
#estandar del mismo 
def desempeno(peso, media_rendimiento, covmatrix, time):
    rendimiento = np.sum(media_rendimiento*peso)*time 
    std = np.sqrt(np.dot(peso.T, np.dot(covmatrix, peso))) * np.sqrt(time)
    return rendimiento, std 

#peso = np.array([0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666])
#peso /= np.sum(peso) #redondeamos los pesos para que sumen 1

#time = 100
#inversion_inicial = 10000
#rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 

#pRet, pStd = desempeno(peso, rendimiento_medio, covmatrix, time)

        
def historicalVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        return np.percentile(rendimiento,alpha)
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.aggregate(historicalVar, alpha=alpha)
    else :
        raise TypeError('Se espera que rendimiento sea dataframe o serie')

#hVaR = -historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
#print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))


#calcula el valor VaR historico condicional (los valores que exceden la medida
#del VaR historico) dado un intervalo de confianza alpha
def historicalCVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVar = rendimiento <= historicalVar(rendimiento, alpha=alpha)
        return rendimiento[belowVar].mean()
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.aggregate(historicalCVar, alpha=alpha)
    else:
        raise TypeError('Se espera que rendimiento sea dataframe o serie')
        
#hCVaR = -historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
#print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))


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


#mc_sims = 400 # numero de simulaciones
#T = 100 #periodo de tiempo en dias 

#portfolio_sims, portResults = MonteCarlo(mc_sims, T, rendimiento_medio, peso, inversion_inicial, covmatrix)

#Mcvar calcula el valor en riesgo historico para los datos optenidos en la
#simulacion montecarlo 
def mcVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        return np.percentile(rendimiento, alpha)
    else:
        raise TypeError("Se espera una serie de datos de pandas")
        
#MCVaR = inversion_inicial - mcVaR(portResults, alpha=5)
#print(" MC VaR  95th CI          :    ", round(MCVaR, 2))


def mcCVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVaR = rendimiento <= mcVaR(rendimiento, alpha=alpha)
        return rendimiento[belowVaR].mean()
    else:
        raise TypeError("Se espera una serie de datos de pandas")
     

#utilizamos las funciones mcvar y mccvar para calcular los valores var
#MCCVaR = inversion_inicial - mcCVaR(portResults, alpha=5)
#print(" MC CVaR 95th CI          :    ", round(MCCVaR, 2))
      

def resum(inversion_inicial, initialPortfolio, hVaR, hCVaR, MCVaR, MCCVaR, pRet):
    print("\nVaR:")

    print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))
    print(" MC VaR  95th CI          :    ", round(MCVaR, 2))

    print("\nCVaR:")

    print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))
    print(" MC CVaR 95th CI          :    ", round(MCCVaR, 2))

    print("\nPortfolio")

    print(' initial portfolio         :    ', round(initialPortfolio))
    print(' portfolio performance     :    ', round(pRet*inversion_inicial))
    
#resum(inversion_inicial, inversion_inicial, hVaR, hCVaR, MCVaR, MCCVaR, pRet)