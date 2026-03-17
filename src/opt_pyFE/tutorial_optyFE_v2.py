# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:45:42 2026

@author: david
"""
import opt_pyFE as opt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import glob

#Definimos los parametros con los que trabajaremos y las especificaciones
#de nuestro analisis 

#--------------------------------------
# El codigo unicamente analziara las acciones que ingresemos y las tomara
# como el portafolio a optimizar,es necesario crear un flujo de seleccion cuyo
# resultado pase a la optimizacion
#--------------------------------------

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
 
#--------------------------------------
# Es necesario determinar la ventana de operacion. Adicionalmente, el codgio
# unicamente funciona con los datos historicos almacenados en
# en una carpeta local
#--------------------------------------

# Fecha de inicio del análisis
start_date = "2023-05-26"  
# Fecha final del análisis
end_date = "2024-05-26"   

# Numero de portafolios que se simularan para la optimizacion
num_portafolios=5000

time = 100 #periodo de tiempo en dias 
inversion_inicial = 10000 #valor del portafolio
mc_sims = 400 # numero de simulaciones

# ubicacion de la carpeta con los bases
path = 'C:/Users/david/Documents/Programación/datos_optpy'


df_wide, df_long = opt.getdata(path, tickers, start_date, end_date)

pendientes = opt.proyeccion(df_long)

datos_bollinger = opt.bandas_bollinger(df_long)

pesos = opt.markowitz(df_wide)

opt.hVaR(df_wide, pesos, time, inversion_inicial)
opt.McVaR(df_wide,mc_sims, time, pesos, inversion_inicial)

