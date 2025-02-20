========
opt_pyFE
========


.. image:: https://img.shields.io/pypi/v/opt_pyFE.svg
        :target: https://pypi.python.org/pypi/opt_pyFE

.. image:: https://img.shields.io/travis/[EDN23,gabrielaadehesa,Dayjingg,Deon9802,alfredo203]/opt_pyFE.svg
        :target: https://travis-ci.com/[EDN23,gabrielaadehesa,Dayjingg,Deon9802,alfredo203]/opt_pyFE

.. image:: https://readthedocs.org/projects/opt-pyFE/badge/?version=latest
        :target: https://opt-pyFE.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




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

#tickers = ["GOOG","BKNG","META", "AAPL","TSLA","^IRX"] #creamos un portafolio

#end_date = dt.datetime.now()
#start_date = end_date - dt.timedelta(days=800)

#rendimiento, rendimiento_medio, covmatrix = getdata(tickers, start_date, end_date)
#rendimiento = rendimiento.dropna() #eliminamos los valores nulos 


* desempeño: Devuelve 2 valor, el rendimiento y la desviación estandar. Requiere una lista de pesos por ticker (que deben sumar 1), la media de rendimiento, una matriz de covarianza y el periodo analizar expresado en días

#peso = np.array([0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666])
#peso /= np.sum(peso) #redondeamos los pesos para que sumen 1

#time = 100
#inversion_inicial = 10000
#rendimiento['portafolio'] = rendimiento.dot(peso) #dot calcula el producto 

#pRet, pStd = desempeno(peso, rendimiento_medio, covmatrix, time)


* historicalVar: Mide el nivel máximo de perdidas que se espera tener con nuestros portafolio, basado en el nivel de precios historicos. Regresa el valor en riesgo historico e imprime su valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

#hVaR = -historicalVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
#print(' historical VaR 95th CI   :    ', round(inversion_inicial*hVaR, 2))


* historicalCVAr: Mide el nivel máximo de perdidas que superan el valor del VaR historico, refleja el nivel de perdidas esperado en el caso más extremo. Regresa el valor en riesgo condicional historico e imprime el valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

#hCVaR = -historicalCVar(rendimiento['portafolio'], alpha=5)*np.sqrt(time)
#print(' historical CVaR 95th CI  :    ', round(inversion_inicial*hCVaR, 2))


* MonteCarlo: Ejecuta una simulación Montecarlo, necesaria para calcular el Var y CVaR por Montecarlo. Requiere el número de simulaciones a ejecutar, el numero de días a simular, la media de rendimiento, la lista de pesos, el valor de nuestro portafolio inicial y la matríz de covarianza

#mc_sims = 400 # numero de simulaciones
#T = 100 #periodo de tiempo en dias 

#portfolio_sims, portResults = MonteCarlo(mc_sims, T, rendimiento_medio, peso, inversion_inicial, covmatrix)


* mcVaR: Mide el nivel máximo de perdidas que se espera tener con nuestros portafolio, basado en el nivel de precios obtenidos en la simulación Montecarlo. Regresa el valor en riesgo por Montecarlo e imprime su valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

#MCVaR = inversion_inicial - mcVaR(portResults, alpha=5)
#print(" MC VaR  95th CI          :    ", round(MCVaR, 2))


* mcCVaR: Mide el nivel máximo de perdidas que superan el valor del VaR Montecarlo, refleja el nivel de perdidas esperado en el caso más extremo. Regresa el valor en riesgo condicional por Montecarlo e imprime el valor en la consola. Requiere el cuadro de rendimientos y un nivel de significancia, alpha

#MCCVaR = inversion_inicial - mcCVaR(portResults, alpha=5)
#print(" MC CVaR 95th CI          :    ", round(MCCVaR, 2))


* resum: Presenta un resumen de los valores VaR y CVar calculados. Requiere de la variable inversion inicial, hVar, hCVar, McVar, McCVar y el rendimiento

#resum(inversion_inicial, inversion_inicial, hVaR, hCVaR, MCVaR, MCCVaR, pRet)


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
