# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 14:41:18 2025

@author: Usuario
"""

#Permite conocer el desempeño de portafolios especificos, por medio del cálculo de rendimientos y la desviación estandar de los tickers

#Gestiona el riesgo del portafolio cálculando el valor en riesgo (VaR) y el valor en riesgo condicional (CVaR) por medio del método histórico y por medio de simulaciones Montecarlo

#Genera un resumen de los valores VaR y CVaR calculados, así como del desempeño del portafolio 
#python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt ###### FALTABA IMPORTAR ESTA
import math
start_date = '2024-01-01'
end_date = '2024-09-23'

#Tutorial

# Función proyección

def proyeccion(tickers, window = 50):
    # Descargar datos del tickers actual
    data = yf.download(tickers, start=start_date, end=end_date)

    # Comprobar si hay datos suficientes
    if not data.empty:
        # Crear DataFrame y agregar columna de días
        df_aj = pd.DataFrame(data)
        df_aj['Día'] = np.arange(1, len(df_aj) + 1)

        # Calcular la media móvil de los últimos 50 días
        df_aj['MA_50'] = df_aj['Close'].rolling(window=window).mean()

        # Variables independientes (Día) y dependientes (Precio de Close)
        X = df_aj[['Día']]  # Días como variable independiente
        y = df_aj['Close']  # Precio de cierre como variable dependiente
        df_aj
     
#Convierte data en un DataFrame llamado df_aj.
#Crea una nueva columna Día que representa el número de día (1, 2, 3, ..., n
#Define las variables para el modelo de regresión: X: La variable independiente (los días). y: La variable dependiente (el precio de cierre).

#regresión
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
        plt.plot(df_aj['Día'], df_aj['Close'], color='blue', label='Precio de Cierre')

        # Gráfico de la media móvil de 50 días
        plt.plot(df_aj['Día'], df_aj['MA_50'], color='green', label='Media Móvil 50 días')
        # Gráfico de la regresión lineal
        plt.plot(df_aj['Día'], y_pred, color='red', label='Línea de Regresión')

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

#Se crea una instancia del modelo de regresión lineal de scikit-learn (LinearRegression()). Se ajusta el modelo con los datos de entrenamiento X (días) y y (precios de cierre). Se usa el modelo entrenado para hacer predicciones (y_pred), es decir, estimar los precios de cierre en función de los días.

#plt.figure(figsize=(10, 6)): Se define el tamaño de la figura.
#plt.plot(df_aj['Día'], df_aj['Close'], color='blue', label='Precio de Cierre'): Grafica la serie de precios de cierre en azul.
#plt.plot(df_aj['Día'], df_aj['MA_50'], color='green', label=f'Media Móvil {window} días'): Grafica la media móvil en verde.
#plt.plot(df_aj['Día'], y_pred, color='red', label='Regresión Lineal'): Grafica la regresión lineal en rojo.
#Título, etiquetas y leyenda: Se agregan título, etiquetas de ejes y leyenda para hacer la gráfica más comprensible.
#plt.grid(True): Activa la cuadrícula para mejorar la lectura de la gráfica. plt.show(): Muestra la gráfica.

#Función calcular_rendimientos_log

def calcular_rendimientos_log(df_aj):
    # Calcula los rendimientos simples porcentuales
    rendimiento = df_aj.pct_change()
    # Convierte los rendimientos simples en rendimientos logarítmicos
    log_returns = np.log(1 + rendimiento)
    # Retorna los rendimientos logarítmicos
    return log_returns

#Usa pct_change(), que calcula el cambio porcentual entre cada período consecutivo de la serie de precios.
#Se transforma el rendimiento simple en rendimiento logarítmico usando la función np.log().
#Esta conversión es útil porque los rendimientos logarítmicos son aditivos en el tiempo, lo que facilita el cálculo de rendimientos acumulados.
#return log_returns: Devuelve los rendimientos logarítmicos como una nueva serie.

#Función simular_portafolios

def simular_portafolios(log_returns, num_portafolios=5000):
    # Obtiene el número de activos en el portafolio
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

    # Retorna las matrices de pesos, rendimientos, volatilidades y ratios Sharpe
    return weight, ReturnEsp, VolEsp, RadioSharpe

#log_returns: DataFrame de Pandas con los rendimientos logarítmicos de los activos del portafolio.
#num_portafolios (opcional, por defecto 5000): Número de portafolios aleatorios a simular.
#Determina el número de activos a partir de la cantidad de columnas en log_returns.
#Inicializa matrices para almacenar pesos de los activos, rendimientos esperados, volatilidades y ratios de Sharpe.
#Calcula la media de los rendimientos logarítmicos (meanlogReturns) y la matriz de covarianza (Sigma).
#Genera num_portafolios portafolios con pesos aleatorios normalizados.
#Para cada portafolio: Calcula el rendimiento esperado como la suma ponderada de los rendimientos medios.
#Calcula la volatilidad esperada con la fórmula de varianza de portafolio.
#Calcula el ratio de Sharpe como la razón entre rendimiento esperado y volatilidad.
#Devuelve las matrices con los pesos, rendimientos, volatilidades y ratios de Sharpe.

#Función encontrar_mejor_portafolio

def encontrar_mejor_portafolio(weight, ReturnEsp, VolEsp, RadioSharpe):
    # Encuentra el índice del portafolio con el mayor Ratio Sharpe
    max_index = RadioSharpe.argmax()
    # Obtiene los pesos, rendimiento, volatilidad y Ratio Sharpe del mejor portafolio
    best_weights = weight[max_index, :]
    return best_weights, ReturnEsp[max_index], VolEsp[max_index], RadioSharpe[max_index]

#weight: Matriz de pesos de los activos en cada portafolio simulado.
#ReturnEsp: Vector con los rendimientos esperados de cada portafolio.
#VolEsp: Vector con las volatilidades esperadas de cada portafolio.
#RadioSharpe: Vector con los Ratios de Sharpe de cada portafolio.
#Encuentra el índice del portafolio con el mayor Ratio de Sharpe usando argmax().
#Extrae los pesos óptimos, rendimiento esperado, volatilidad y Ratio de Sharpe del mejor portafolio.
#Retorna estos valores como una tupla. Salida (return values) best_weights: Pesos del portafolio con el mejor Ratio de Sharpe.
#ReturnEsp[max_index]: Rendimiento esperado del mejor portafolio.
#VolEsp[max_index]: Volatilidad del mejor portafolio.
#RadioSharpe[max_index]: Ratio de Sharpe del mejor portafolio.

#Funcion mostrar_resultados

def mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio):
    # Muestra los pesos de los activos en el mejor portafolio
    print("Mejores pesos del portafolio:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {best_weights[i] * 100:.2f}%")

    # Muestra el rendimiento esperado, la volatilidad y el Ratio Sharpe del mejor portafolio
    print(f"Retorno esperado del portafolio: {retorno:.4f}")
    print(f"Volatilidad esperada del portafolio: {volatilidad:.4f}")
    print(f"Ratio Sharpe máximo: {sharpe_ratio:.4f}")

#tickers: Lista con los nombres o símbolos de los activos en el portafolio.
#best_weights: Vector con los pesos asignados a cada activo en el mejor portafolio.
#retorno: Rendimiento esperado del mejor portafolio.
#volatilidad: Volatilidad esperada del mejor portafolio.
#sharpe_ratio: Ratio de Sharpe del mejor portafolio.
#Imprime los pesos de cada activo en el portafolio en formato porcentual.
#Muestra los valores clave del portafolio: Rendimiento esperado (esperanza de retorno del portafolio).
#Volatilidad esperada (riesgo medido como desviación estándar).
#Ratio de Sharpe (rendimiento ajustado al riesgo).

#get data: Recupera datos historicos de las emisoras alojadas en el registro de Yahoo Finance. Devuelve una tabla con la información solicitada, una tabla con los rendimientos de las emisoras, la media de los rendimientos y una matriz de covarianza. Requiere una lista con los tickers de las emisoras, así como fechas de inicio y final del periodo a trabajar

tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #creamos un portafolio

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=800)

def getdata(stocks, start, end): 
    stockdata =yf.download(stocks, start=start, end=end) 
    stockdata = stockdata['Close'] 
    rendimiento = stockdata.pct_change() 
    media_rendimiento = rendimiento.mean()
    covmatrix = rendimiento.cov()
    return rendimiento, media_rendimiento, covmatrix 

rendimiento, rendimiento_medio, covmatrix = getdata(tickers, start_date, end_date)
rendimiento = rendimiento.dropna() #eliminamos los valores nulos 


#desempeno: Devuelve 2 valor, el rendimiento y la desviación estandar. Requiere una lista de pesos por ticker (que deben sumar 1), la media de rendimiento, una matriz de covarianza y el periodo analizar expresado en días
def desempeno(peso, media_rendimiento, covmatrix, time):
    rendimiento = np.sum(media_rendimiento*peso)*time 
    std = np.sqrt(np.dot(peso.T, np.dot(covmatrix, peso))) * np.sqrt(time)
    return rendimiento, std 

peso = np.array([0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666])
peso /= np.sum(peso) #redondeamos los pesos para que sumen 1

time = 100
inversion_inicial = 10000
rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 

pRet, pStd = desempeno(peso, rendimiento_medio, covmatrix, time)
#############

#historicalVar: Mide el nivel máximo de perdidas que se espera tener con nuestros portafolio, basado en el nivel de precios historicos. Regresa el valor en riesgo historico e imprime su valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

def historicalVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        return np.percentile(rendimiento,alpha)
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.aggregate(historicalVar, alpha=alpha)
    else :
        raise TypeError('Se espera que rendimiento sea dataframe o serie')

hVaR = -historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))


#historicalCVAr: Mide el nivel máximo de perdidas que superan el valor del VaR historico, refleja el nivel de perdidas esperado en el caso más extremo. Regresa el valor en riesgo condicional historico e imprime el valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha
def historicalCVar(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVar = rendimiento <= historicalVar(rendimiento, alpha=alpha)
        return rendimiento[belowVar].mean()
    elif isinstance(rendimiento, pd.DataFrame):
        return rendimiento.aggregate(historicalCVar, alpha=alpha)
    else:
        raise TypeError('Se espera que rendimiento sea dataframe o serie')

hCVaR = -historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))


#MonteCarlo: Ejecuta una simulación Montecarlo, necesaria para calcular el Var y CVaR por Montecarlo. Requiere el número de simulaciones a ejecutar, el numero de días a simular, la media de rendimiento, la lista de pesos, el valor de nuestro portafolio inicial y la matríz de covarianza
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

mc_sims = 400 # numero de simulaciones
T = 100 #periodo de tiempo en dias 

portfolio_sims, portResults = MonteCarlo(mc_sims, T, rendimiento_medio, peso, inversion_inicial, covmatrix)


#mcVaR: Mide el nivel máximo de perdidas que se espera tener con nuestros portafolio, basado en el nivel de precios obtenidos en la simulación Montecarlo. Regresa el valor en riesgo por Montecarlo e imprime su valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha
def mcVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        return np.percentile(rendimiento, alpha)
    else:
        raise TypeError("Se espera una serie de datos de pandas")

MCVaR = inversion_inicial - mcVaR(portResults, alpha=5)
print(" MC VaR  95th CI          :    ", round(MCVaR, 2))


#mcCVaR: Mide el nivel máximo de perdidas que superan el valor del VaR Montecarlo, refleja el nivel de perdidas esperado en el caso más extremo. Regresa el valor en riesgo condicional por Montecarlo e imprime el valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha
def mcCVaR(rendimiento, alpha=5):
    if isinstance(rendimiento, pd.Series):
        belowVaR = rendimiento <= mcVaR(rendimiento, alpha=alpha)
        return rendimiento[belowVaR].mean()
    else:
        raise TypeError("Se espera una serie de datos de pandas")

MCCVaR = inversion_inicial - mcCVaR(portResults, alpha=5)
print(" MC CVaR 95th CI          :    ", round(MCCVaR, 2))


#resum: Presenta un resumen de los valores VaR y CVar calculados. Requiere de la variable inversion inicial, hVar, hCVar, McVar, McCVar y el rendimiento
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

resum(inversion_inicial, inversion_inicial, hVaR, hCVaR, MCVaR, MCCVaR, pRet)
# Tutorial Analíticos: Proyección de Acciones con Python

## 1. Importar las librerías

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#fechas
start_date = '2024-01-01'
end_date = '2024-09-23'

## 3. Función `proyeccion`
#La función realiza:
#Descarga de datos**:
#Usa `yfinance` para obtener precios históricos.
  #Filtra por `start_date` y `end_date`.
#- **Cálculos**:
    #Media móvil (default: 50 días).
  #- Regresión lineal con `scikit-learn`.
     #Gráficos**:
  #Muestra: precio de cierre, media móvil y tendencia.

## 4. Ejemplo de uso
#Analizar múltiples tickers:

# Definir los tickers a analizar
tickers = ["AAPL", "GOOG", "BKNG"] 

# Bucle para procesar cada ticker
for ticker in tickers:
    proyeccion(ticker)


#Cambiar la ventana de la media móvil (ej: 20 días):


for ticker in tickers:
    proyeccion(ticker, window=20)  # Media móvil de 20 días

#intento de histograma de precios finales 

precios_finales=portfolio_sims[-1, :]
R=np.max(precios_finales)-np.min(precios_finales) #rango
k=int(np.ceil(np.log2(R)+1))
#k es numero de intervalos
inf = precios_finales.min()        # Limite inferior del primer intervalo
dif = precios_finales.max()
sup = precios_finales.max() + 1    # Limite superior del último intervalo

intervals = pd.interval_range(
    start=inf,
    end=sup,
    periods=k,
    name="Intervalo",
    closed="left")

df1 = pd.DataFrame(index=intervals)
df1["FreqAbs"] = pd.cut(precios_finales, bins=df1.index).value_counts()
w=df1["FreqAbs"]
fr=[f/400 for f in w]

x= np.arange(len(intervals))
plt.bar(x, fr, width=0.8, edgecolor="black",
        color="lightgreen")

plt.xticks(x,intervals, rotation=45)
plt.title("histograma")
plt.xlabel("intervalos")
plt.ylabel("frecuencia relativa")
plt.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()
#


#histograma por trayectoria
Troubleshooting
--------


* Problema con descarga de datos de yfinanca
Fecha: 21/02/2025
Versión:
Realizado por: David Gutiérrez

yfinance importaba datos nulos o no descargaba la información solicitada. Este problema apareció después de no utilizar la libreria durante un periodo de tiempo prolongado e impedía el obtener cualqier tipo de dato, para cualquier periodo, de yfinance.

Se detectó que yfinance había sido actualizada durante este periodo de tiempo, tras comprobar que ninguna otra alteranitva de sintaxis obtenía resultados diferentes se procedió a actualizar la libría de manera manual, por medio del comando: "pip install --upgrade yfinance"
Este deberá ser ingresado en el Anaconda Prompt del equipo. 

A pesar de que la actulización fue instalada con éxito el problema persistió, por lo que procedimos a verificar si había algún problema con el estado de los permisos JSON. Por medio del siguiente código revisamos si yfinance estaba devolviendo datos o si el problema era de otra naturaleza:
```python
import requests

url = "https://query1.finance.yahoo.com/v8/finance/chart/MSFT"
response = requests.get(url)
print(response.text)  # If empty or malformed, Yahoo API might be down
```

La respuesta del programa fue: "Edge: Too Many Requests". Esto nos indica que yfinnace está bloqueando nuestra dirección IP de manera temporal debido a un alto número de solicitudes enviadas.
