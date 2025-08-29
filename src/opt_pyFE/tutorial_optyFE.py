# -*- coding: utf-8 -*-
"""
Created on Tue May 13 13:58:35 2025

@author: david
"""
#importamos el paquete de optimización de portafolios
import opt_pyFE as opt
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


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
start_date = "2024-05-26"  
# Fecha final del análisis
end_date = "2025-05-26"   

#MODULO DE ANALITICOS#

#Generamos un cilo for para graficar la media movil y la tendencia de cada
#uno de nuestros tickers 
for ticker in tickers:
    opt.proyeccion(tickers, start_date, end_date)

#La media movil se calcula por defecto a 50 días pero es posible ajustar
#este parametro especificando el valor de window 
for ticker in tickers:
    opt.proyeccion(tickers, start_date, end_date, window = 20)
    
#Se genera un grafico de bandas de bollinger por cada una de los tickers
opt.bandas_bollinger(tickers, start_date, end_date)


#MODULO DE OPTIMIZACION#

#Se descargan y tratan los datos para simular el peso de los portafolios
df_aj = opt.descargar_datos(tickers, start_date, end_date) 

#se calcula el rendimiento logaritmico de las emisoras
log_returns = opt.calcular_rendimientos_log(df_aj) ## @David esto crea Warning revisar

#Se simula un numero definido de portafolios para poder encontrar el optimo
num_portafolios = 10000  
weight, ReturnEsp, VolEsp, RadioSharpe = opt.simular_portafolios(log_returns, 
                                                             num_portafolios)

#Se encuentra el portofalio que miniza el riesgo entre las simulaciones
#tomando esta como la distribución de pesos optima
best_weights, retorno, volatilidad, sharpe_ratio = opt.encontrar_mejor_portafolio(weight, 
                                                                              ReturnEsp, 
                                                                              VolEsp, 
                                                                              RadioSharpe)

#Se imprime en la consola un resumen de los resultados obtenidos y
#de los pesos seleccionados
opt.mostrar_resultados(tickers, 
                   best_weights, 
                   retorno, 
                   volatilidad, 
                   sharpe_ratio)

#MODULO DE RIESGO#

#Calculamos los rendimientos y geenramos una matriz de covarianza 
#para poder calcular los riesgo de nuestro portafolio
rendimiento, rendimiento_medio, covmatrix = opt.getdata(tickers, 
                                                    start_date, 
                                                    end_date)
#retiramos los valores nulos de nuestro datafreame 
rendimiento = rendimiento.dropna() #eliminamos los valores nulos 

#creamos un array de numpy para almacenar los pesos optimos 
peso = np.array([0.0104, 0.1213, 0.0562, 0.1209, 0.0576, 0.0642, 0.0061, 
                 0.0222, 0.0973, 0.1048, 0.0751, 0.0628, 0.0354, 0.0925, 0.0732
])
peso /= np.sum(peso) #redondeamos los pesos para que sumen 1


time = 100 #Definimos el numero de dias 
inversion_inicial = 100000 #Definimos el monto a invertir 
rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 

#Calculamos el rendimiento esperado y la desviacion estandar anualizada 
pRet, pStd = opt.desempeno(peso, 
                       rendimiento_medio, 
                       covmatrix, 
                       time)

#Con los parametros obtenidos de los datos historicos calculamos el Var
hVaR = -opt.historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))

#Con los parametros obtenidos de los datos historicos calculamos el Var condicional
hCVaR = -opt.historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))


mc_sims = 1000 # numero de simulaciones
T = 100 #periodo de tiempo en dias 

#Se generan las simulaciones montecarlo 
portfolio_sims, portResults = opt.MonteCarlo(mc_sims, 
                                         T, 
                                         rendimiento_medio, 
                                         peso, inversion_inicial, 
                                         covmatrix)

#Con los parametros obtenidos de los datos historicos calculamos el Var
MCVaR = inversion_inicial - opt.mcVaR(portResults, alpha=5)
print(" MC VaR  95th CI          :    ", round(MCVaR, 2))

#Con los parametros obtenidos de los datos historicos calculamos el Var condicional
MCCVaR = inversion_inicial - opt.mcCVaR(portResults, alpha=5)
print(" MC CVaR 95th CI          :    ", round(MCCVaR, 2))

#Se imprime un resumen de los valores de todos los valores calculados
opt.resum(inversion_inicial, inversion_inicial, hVaR, hCVaR, MCVaR, MCCVaR, pRet)
