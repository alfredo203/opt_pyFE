
# opt_pyFE

Este proyecto es una optimización de portafolio utilizando información extraída de yahoo finance.

"Potencia tus inversiones con nuestro avanzado paquete de optimización de portafolio."  Nuestra herramienta, basada en algoritmos sofisticados, analiza tu perfil de riesgo y te ayuda a construir una cartera diversificada que maximice tus rendimientos y minimice tu exposición a pérdidas.  Con seguimiento en tiempo real y rebalanceos automáticos, tendrás la tranquilidad de saber que tu dinero está trabajando inteligentemente para ti. ¡Toma el control de tu futuro financiero hoy mismo!



* Free software: MIT license
* Documentation: https://opt-pyFE.readthedocs.io.


Features
--------

* Permite conocer el desempeño de portafolios especificos, por medio del cálculo de rendimientos y la desviación estandar de los tickers

* Gestiona el riesgo del portafolio cálculando el valor en riesgo (VaR) y el valor en riesgo condicional (CVaR) por medio del método histórico y por medio de simulaciones Montecarlo

* Genera un resumen de los valores VaR y CVaR calculados, así como del desempeño del portafolio 

Tutorial
--------

* get data: Recupera datos historicos de las emisoras alojadas en el registro de Yahoo Finance. Devuelve una tabla con la información solicitada, una tabla con los rendimientos de las emisoras, la media de los rendimientos y una matriz de covarianza. Requiere una lista con los tickers de las emisoras, así como fechas de inicio y final del periodo a trabajar
```python
tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #creamos un portafolio

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=800)

rendimiento, rendimiento_medio, covmatrix = getdata(tickers, start_date, end_date)
rendimiento = rendimiento.dropna() #eliminamos los valores nulos 
```

* desempeno: Devuelve 2 valor, el rendimiento y la desviación estandar. Requiere una lista de pesos por ticker (que deben sumar 1), la media de rendimiento, una matriz de covarianza y el periodo analizar expresado en días

```python
peso = np.array([0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666])
peso /= np.sum(peso) #redondeamos los pesos para que sumen 1

time = 100
inversion_inicial = 10000
rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 

pRet, pStd = desempeno(peso, rendimiento_medio, covmatrix, time)
```

* historicalVar: Mide el nivel máximo de perdidas que se espera tener con nuestros portafolio, basado en el nivel de precios historicos. Regresa el valor en riesgo historico e imprime su valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

```python
hVaR = -historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))
```

* historicalCVAr: Mide el nivel máximo de perdidas que superan el valor del VaR historico, refleja el nivel de perdidas esperado en el caso más extremo. Regresa el valor en riesgo condicional historico e imprime el valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

```python
hCVaR = -historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))
```

* MonteCarlo: Ejecuta una simulación Montecarlo, necesaria para calcular el Var y CVaR por Montecarlo. Requiere el número de simulaciones a ejecutar, el numero de días a simular, la media de rendimiento, la lista de pesos, el valor de nuestro portafolio inicial y la matríz de covarianza

```python
mc_sims = 400 # numero de simulaciones
T = 100 #periodo de tiempo en dias 

portfolio_sims, portResults = MonteCarlo(mc_sims, T, rendimiento_medio, peso, inversion_inicial, covmatrix)
```

* mcVaR: Mide el nivel máximo de perdidas que se espera tener con nuestros portafolio, basado en el nivel de precios obtenidos en la simulación Montecarlo. Regresa el valor en riesgo por Montecarlo e imprime su valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

```python
MCVaR = inversion_inicial - mcVaR(portResults, alpha=5)
print(" MC VaR  95th CI          :    ", round(MCVaR, 2))
```

* mcCVaR: Mide el nivel máximo de perdidas que superan el valor del VaR Montecarlo, refleja el nivel de perdidas esperado en el caso más extremo. Regresa el valor en riesgo condicional por Montecarlo e imprime el valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

```python
MCCVaR = inversion_inicial - mcCVaR(portResults, alpha=5)
print(" MC CVaR 95th CI          :    ", round(MCCVaR, 2))
```

* resum: Presenta un resumen de los valores VaR y CVar calculados. Requiere de la variable inversion inicial, hVar, hCVar, McVar, McCVar y el rendimiento

```python
resum(inversion_inicial, inversion_inicial, hVaR, hCVaR, MCVaR, MCCVaR, pRet)
```

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
La solución a este problema fue cambiar de red. Aunque, de ser posible, reiniciar la red o usar una VPN podrían funcionar en este tipo de casos.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
