# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:51:29 2024

@author: edson, gayal, David, david

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

tickers = ["AA","AAL","AAPL","AMM.TO","ABBV","ABNB","ACTINVRB.MX","AC","AFRM",
           "AGNC","ALFAA.MX","ALPEKA.MX","ALSEA.MX","AMAT","AMD","AMX","AMZN",
           "APA","ASURB.MX","ATER","ATOS","AIY.DE","AVGO","AXP","BABA","BAC",
           "BA","BBAJIOO.MX","BIMBOA.MX","BMY","BNGO","CAT","CCL",
           "CEMEXCPO.MX","CHDRAUIB.MX","CLF","COST","CRM","CSCO",
           "CUERVO.MX","CVS","CVX","C","DAL","DIS","DVN","ELEKTRA.MX","ETSY",
           "FANG","FCX","FDX","FEMSAUBD.MX","FIBRAMQ12.MX","FIBRAPL14.MX",
           "FSLR","FUBO","FUBO","FUNO11.MX","F","GAPB.MX","GCARSOA1.MX","GCC",
           "GENTERA.MX","GE","GFINBURO.MX","GFNORTEO.MX","GILD","GMEXICOB.MX",
           "GME","GM","GOLD","GOOGL","GRUMAB.MX","HD","INTC","JNJ","JPM",
           "KIMBERA.MX","KOFUBL.MX","KO","LABB.MX",
          "LASITEB-1.MX","LCID","LIVEPOLC-1.MX","LLY","LUV","LVS","LYFT","MARA",
          "MARA","MA","MCD","MEGACPO.MX","MELIN.MX","META","MFRISCOA-1.MX","MGM",
          "MRK","MRNA","MRO","MSFT","MU","NCLHN.MX","NFLX","NKE","NKLA","NUN.MX",
          "NVAX","NVDA","OMAB.MX","ORBIA.MX","ORCL","OXY1.MX","PARA","PBRN.MX","PE&OLES.MX",
          "PEP","PFE","PG","PINFRA.MX","PINS","PLTR","PYPL","QCOM","Q.MX","RCL",
          "RIOT","RIVN","ROKU","RA.MX","SBUX","SHOP","SITES1A-1.MX","SKLZ",
          "SOFI","SPCE","SQ","TALN.MX","TERRA13.MX","TGT","TLEVISACPO.MX","TMO",
          "TSLA","TSMN.MX","TWLO","TX","T","UAL","UBER","UNH","UPST","VESTA.MX",
          "VOLARA.MX","VZ","V","WALMEX.MX","WFC","WMT","WYNN","XOM","X","ZM"] 

# Primera versión de proyección: 
# def proyección(ticker):
#     # Descargar datos del ticker actual
#     data = yf.download(ticker, start=start_date, end=end_date)
    
#     # Comprobar si hay datos suficientes
#     if not data.empty:
#         # Crear DataFrame y agregar columna de días
#         df_aj = pd.DataFrame(data)
#         df_aj['Día'] = np.arange(1, len(df_aj) + 1)
        
#         # Calcular la media móvil de los últimos 50 días
#         df_aj['MA_50'] = df_aj['Close'].rolling(window=20).mean()

#         # Imprimir la tabla con las columnas relevantes
#         print(f"\nTabla de Precios y Media Movil de 50 días para {ticker}:\n")
#         print(df_aj[['Día', 'Close', 'MA_50']].tail(40))  # Mostrar las últimas 40 filas
#     else:
#         print(f"No se encontraron datos para {ticker}")

# Sobrescribir clean_data: ejecutar regresión y generar gráficos
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

# Función para descargar los datos de cierre ajustado de los tickers seleccionados
def descargar_datos(tickers, start_date, end_date, datos_req=["Close"]):
    # Descarga los datos de Yahoo Finance en el rango de fechas especificado
    df_aj = yf.download(tickers, start=start_date, end=end_date)
    # Devuelve solo las columnas solicitadas (por defecto, el precio de cierre ajustado)
    return df_aj
    # Calcula los rendimientos simples porcentuales

# Función para calcular los rendimientos logarítmicos
    # df_aj = descargar_datos(tickers, start_date, end_date)
def calcular_rendimientos_log(df_aj):
    # Calcula los rendimientos simples porcentuales    
    rendimiento = df_aj.pct_change()
    # Convierte los rendimientos simples en rendimientos logarítmicos
    log_returns = np.log(1 + rendimiento)
    # Retorna los rendimientos logarítmicos
    return log_returns
    
# Función para simular múltiples portafolios
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

# Función para identificar el mejor portafolio según el Ratio Sharpe
def encontrar_mejor_portafolio(weight, ReturnEsp, VolEsp, RadioSharpe):
    # Encuentra el índice del portafolio con el mayor Ratio Sharpe
    max_index = RadioSharpe.argmax()
    # Obtiene los pesos, rendimiento, volatilidad y Ratio Sharpe del mejor portafolio
    best_weights = weight[max_index, :]
    return best_weights, ReturnEsp[max_index], VolEsp[max_index], RadioSharpe[max_index]

# Función para mostrar los resultados del mejor portafolio
def mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio):
    # Muestra los pesos de los activos en el mejor portafolio
    print("Mejores pesos del portafolio:")
    for i, ticker in enumerate(tickers):
        print(f"{ticker}: {best_weights[i] * 100:.2f}%")

    # Muestra el rendimiento esperado, la volatilidad y el Ratio Sharpe del mejor portafolio
    print(f"Retorno esperado del portafolio: {retorno:.4f}")
    print(f"Volatilidad esperada del portafolio: {volatilidad:.4f}")
    print(f"Ratio Sharpe máximo: {sharpe_ratio:.4f}")

# Función principal que ejecuta todo el análisis de portafolios
def ejecutar_analisis(tickers, start_date, end_date):
    # Descarga los datos de los tickers seleccionados en el rango de fechas indicado
    df_aj = descargar_datos(tickers, start_date, end_date)
    # Calcula los rendimientos logarítmicos a partir de los precios de cierre ajustados
    log_returns = calcular_rendimientos_log(df_aj)
    # Simula portafolios aleatorios basados en los rendimientos logarítmicos
    weight, ReturnEsp, VolEsp, RadioSharpe = simular_portafolios(log_returns)
    # Encuentra el portafolio con el mayor Ratio Sharpe
    best_weights, retorno, volatilidad, sharpe_ratio = encontrar_mejor_portafolio(weight, ReturnEsp, VolEsp, RadioSharpe)
    # Muestra los resultados del mejor portafolio
    mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio)

# Ejecuta el análisis de portafolios con los tickers y fechas indicados
ejecutar_analisis(tickers, start_date, end_date)


# EJEMPLO DE USO

# Lista de los tickers
tickers = ["AAPL", "MSFT", "AMZN"]  
# Fecha de start_date del análisis
start_date = "2020-01-01"  
# Fecha end_date del análisis
end_date = "2021-01-01"   

df_aj = descargar_datos(tickers, start_date, end_date)

# Ejemplo de uso de la función calcular_rendimientos_log
log_returns = calcular_rendimientos_log(df_aj)

# Ejemplo de uso de la función simular_portafolios
num_portafolios = 5000  
weight, ReturnEsp, VolEsp, RadioSharpe = simular_portafolios(log_returns, num_portafolios)

# Ejemplo de uso de la función encontrar_mejor_portafolio
best_weights, retorno, volatilidad, sharpe_ratio = encontrar_mejor_portafolio(weight, ReturnEsp, VolEsp, RadioSharpe)

# Ejemplo de uso de la función mostrar_resultados
mostrar_resultados(tickers, best_weights, retorno, volatilidad, sharpe_ratio)

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