# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:50:52 2024

@author: Titiru (alfredo203 comenta 24/Oct/2024)
"""

#

import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


def descargar_datos_tickers(tickers, inicio, final, datos_req=["Close"]):
    """
    descarga los datos de los tikers que se quieran utilizar, inicio y final son los parametros de la descarga y los datos requeridos
    puedes escogerse aunque puse por defecto el Close pr que es lo que utilizamos 
    """
    # descargar los datos
    df = yf.download(tickers, start=inicio, end=final)
    
    # seleccionar solo los datos requeridos (como 'Close')
    df_aj = df[datos_req]
    
    # eliminar el nivel superior de la columna ('Close')
    df_aj.columns = df_aj.columns.droplevel(0)
    
    return df_aj


def modelo_match(tickers, inicio, final, epochs=45, batch_size=33, neuronas=50, time_step=1):
    """
    utiliza la funcion anterior para jalar los datos, entrena por cada uno de los tikers el modelo y te da
    una prediccion de 15 dias en el futuro.
  
    """
    # descargar los datos 
    df_aj = descargar_datos_tickers(tickers, inicio, final, datos_req=["Close"])
    
    # almacenar las predicciones de cada ticker
    predicciones_dict = {}

    # hacer el proceso para cada ticker
    for ticker in tickers:
        print(f"Procesando ticker: {ticker}")
        
        # obtener los datos de cierre para el ticker actual
        datos_ticker = df_aj[[ticker]].dropna()

        # crear set de entrenamiento y validación
        set_entrenamiento = datos_ticker[:'2024-10-06']  # ajusta la fecha para el set de entrenamiento
        set_validacion = datos_ticker[:'2024-10-01':]  # ajusta la fecha para el set de validación
        #me gustaria hacerlo para que fuera siempre un mes antes de la fecha actual pero no se como

        # normalizar los valores
        sc = MinMaxScaler(feature_range=(0,1))
        set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

        # crear bloques de datos de entrenamiento
        x_tren, y_tren = [], []
        m = len(set_entrenamiento_escalado)
        
        for i in range(time_step, m):
            x_tren.append(set_entrenamiento_escalado[i-time_step:i, 0])
            y_tren.append(set_entrenamiento_escalado[i, 0])

        x_tren, y_tren = np.array(x_tren), np.array(y_tren)

        # tranfrmamos para que sea compatible con el modelo 
        x_tren = np.reshape(x_tren, (x_tren.shape[0], x_tren.shape[1], 1))

        # crear el modelo LSTM
        modelo = Sequential()
        modelo.add(LSTM(units=neuronas, input_shape=(x_tren.shape[1], 1)))
        modelo.add(Dense(units=1))

        # compilar el modelo
        modelo.compile(optimizer='rmsprop', loss='mse')
        #tengo entendido que hay otras forma de optimizarlo pero no encuentro informacion al respeto 

        # entrenar el modelo
        modelo.fit(x_tren, y_tren, epochs=epochs, batch_size=batch_size)

        # preparar los datos de validación
        x_test = sc.transform(set_validacion)
        X_test = []
        for i in range(time_step, len(x_test)):
            X_test.append(x_test[i-time_step:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # predicción en el set de validación
        prediccion = modelo.predict(X_test)
        prediccion = sc.inverse_transform(prediccion)

        # graficar los resultados de la validación
        plt.figure(figsize=(10, 6))
        plt.plot(set_validacion.values[time_step:], color='blue', label='Valores Reales')
        plt.plot(prediccion, color='red', label='Valores Predichos')
        plt.title(f'Validación del Modelo para {ticker}')
        plt.xlabel('Días')
        plt.ylabel('Precio de la Acción')
        plt.legend()
        plt.show()

        # ultimo bloque para predecir los próximos 15 días
        ultimo_bloque = set_validacion.values[-time_step:]
        ultimo_bloque_escalado = sc.transform(ultimo_bloque)

        # lista para las predicciones futuras
        predicciones_futuras = []

        for i in range(15):
            x_input = np.reshape(ultimo_bloque_escalado, (1, time_step, 1))
            prediccion = modelo.predict(x_input)
            prediccion_invertida = sc.inverse_transform(prediccion)
            predicciones_futuras.append(prediccion_invertida[0, 0])

            nueva_prediccion_escalada = sc.transform(prediccion_invertida)
            ultimo_bloque_escalado = np.append(ultimo_bloque_escalado[1:], nueva_prediccion_escalada).reshape(time_step, 1)

        # convertir a array
        predicciones_futuras = np.array(predicciones_futuras)

        # guardar la predicciones 
        predicciones_dict[ticker] = predicciones_futuras

        # graficar la predicción de los próximos 15 días para este ticker
        plt.figure(figsize=(10, 6))
        plt.plot(predicciones_futuras, color='green', label=f'Predicciones Futuras ({ticker}) - 15 días')
        plt.title(f'Predicción de Precios de {ticker} para los Próximos 15 Días')
        plt.xlabel('Días')
        plt.ylabel('Precio de la Acción')
        plt.legend()
        plt.show()

    return predicciones_dict

# Ejemplo de uso 
# tickers = ["AA","AAL","AAPL","AMM.TO","ABBV","ABNB","ACTINVRB.MX","AC","AFRM",
#            "AGNC","ALFAA.MX","ALPEKA.MX","ALSEA.MX","AMAT","AMD","AMX","AMZN",
#            "APA","ASURB.MX","ATER","ATOS","AIY.DE","AVGO","AXP","BABA","BAC",
#            "BA","BBAJIOO.MX","BIMBOA.MX","BMY","BNGO","CAT","CCL",
#            "CEMEXCPO.MX","CHDRAUIB.MX","CLF","COST","CRM","CSCO",
#            "CUERVO.MX","CVS","CVX","C","DAL","DIS","DVN", "ETSY",
#            "FANG","FCX","FDX","FEMSAUBD.MX","FIBRAMQ12.MX","FIBRAPL14.MX",
#            "FSLR","FUBO","FUBO","FUNO11.MX","F","GAPB.MX","GCARSOA1.MX","GCC",
#            "GENTERA.MX","GE","GFINBURO.MX","GFNORTEO.MX","GILD","GMEXICOB.MX",
#            "GME","GM","GOLD","GOOGL","GRUMAB.MX","HD","INTC","JNJ","JPM",
#            "KIMBERA.MX","KOFUBL.MX","KO","LABB.MX",
#           "LASITEB-1.MX","LCID","LIVEPOLC-1.MX","LLY","LUV","LVS","LYFT","MARA",
#           "MARA","MA","MCD","MEGACPO.MX","MELIN.MX","META","MFRISCOA-1.MX","MGM",
#           "MRK","MRNA","MRO","MSFT","MU","NCLHN.MX","NFLX","NKE","NKLA","NUN.MX",
#           "NVAX","NVDA","OMAB.MX","ORBIA.MX","ORCL","OXY1.MX","PARA","PBRN.MX","PE&OLES.MX",
#           "PEP","PFE","PG","PINFRA.MX","PINS","PLTR","PYPL","QCOM","Q.MX","RCL",
#           "RIOT","RIVN","ROKU","RA.MX","SBUX","SHOP","SITES1A-1.MX","SKLZ",
#           "SOFI","SPCE","SQ","TALN.MX","TERRA13.MX","TGT","TMO",
#           "TSLA","TSMN.MX","TWLO","TX","T","UAL","UBER","UNH","UPST","VESTA.MX",
#           "VOLARA.MX","VZ","V","WALMEX.MX","WFC","WMT","WYNN","XOM","X","ZM"]
# inicio = "2022-03-01"
# final = "2024-10-04"
# predicciones_multiples = modelo_match(tickers, inicio, final)



