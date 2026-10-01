# opt_pyFE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**opt_pyFE** es una librería de Python para análisis financiero cuantitativo, optimización de carteras bajo la Teoría Moderna de Portafolios (Markowitz), modelado de riesgo mediante Métodos Históricos y Simulación Monte Carlo, y analítica de sentimiento con procesamiento de lenguaje natural (NLP).

---

## Características Principales

* **Ingesta de Datos Financieros:** Descarga automatizada y homogeneización de precios históricos y rendimientos continuos o discretos vía Yahoo Finance (`yfinance`).
* **Analítica Técnica:**
  * Proyección de tendencia mediante regresión lineal con medias móviles configurables.
  * Bandas de Bollinger para análisis de volatilidad y reversión a la media.
* **Optimización de Portafolios:**
  * Simulación estocástica multivariada de carteras.
  * Identificación del portafolio óptimo (máximo Ratio de Sharpe) y de mínima volatilidad (mínima varianza).
  * Evaluación de retorno esperado y volatilidad del portafolio en cualquier horizonte temporal.
* **Gestión de Riesgo:**
  * Cálculo de Valor en Riesgo (**VaR**) y Valor en Riesgo Condicional (**CVaR** / Expected Shortfall) por el método histórico.
  * Simulación Monte Carlo multivariada con descomposición de Cholesky y regularización espectral robusta.
  * Estimación de VaR y CVaR sobre las trayectorias proyectadas de Monte Carlo.
  * Reporte tabular y resumido de métricas de riesgo.
* **Procesamiento de Lenguaje Natural (NLP):**
  * Clasificación de sentimiento en titulares financieros con modelos basados en Transformer (`FinancialBERT`).
* **Línea de Comandos (CLI):** Comando `opt-pyfe` para optimizaciones rápidas directamente desde la terminal.

---

## Instalación

### Instalación estándar:
Clona el repositorio e instala en modo editable o directo:
```bash
git clone https://github.com/alfredo203/opt_pyFE.git
cd opt_pyFE
pip install .
```

O para desarrollo:
```bash
pip install -e .
```

### Con soporte para NLP (FinancialBERT):
```bash
pip install ".[nlp]"
```

---

## Estructura del Proyecto

```text
opt_pyFE/
├── pyproject.toml               # Configuración moderna de empaquetado (PEP 518/621)
├── README.md                    # Documentación principal
├── src/
│   └── opt_pyFE/
│       ├── __init__.py          # API pública limpia y controlada (__all__)
│       ├── data.py              # Descarga de datos y rendimientos
│       ├── analytics.py         # Medias móviles, regresión y Bollinger
│       ├── portfolio.py         # Simulación de portafolios y Sharpe
│       ├── risk.py              # VaR, CVaR y simulación Monte Carlo
│       ├── sentiment.py         # Análisis de sentimiento con FinancialBERT
│       ├── cli.py               # Herramienta de línea de comandos (Typer/Rich)
│       └── opt_pyFE.py          # Módulo unificado para retrocompatibilidad
├── examples/
│   ├── tutorial_opt_pyFE.py     # Tutorial completo ejecutable de inicio a fin
│   └── bert_sentiment_demo.py   # Demo de análisis de sentimiento
└── tests/                       # Suite de pruebas unitarias con pytest
    ├── test_data.py
    ├── test_analytics.py
    ├── test_portfolio.py
    ├── test_risk.py
    └── test_imports.py
```

---

## Guía Rápida / Tutorial

Puedes ejecutar el script interactivo completo con:
```bash
python examples/tutorial_opt_pyFE.py
```

O utilizar las funciones directamente en tus propios scripts o notebooks:

### 1. Ingesta de datos y cálculo de rendimientos
```python
import opt_pyFE as opt

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
start_date = "2023-01-01"
end_date = "2025-01-01"

# Descarga de precios de cierre
precios = opt.descargar_datos(tickers, start_date=start_date, end_date=end_date)

# Rendimientos logarítmicos
rendimientos_log = opt.calcular_rendimientos_log(precios)

# Rendimientos simples, media diaria y matriz de covarianza
rendimientos, media_rend, covmatrix = opt.getdata(tickers, start=start_date, end=end_date)
```

### 2. Analítica técnica (Regresión y Bandas de Bollinger)
```python
# Proyección con regresión lineal y media móvil (sin bloquear ejecución obligada)
pendientes = opt.proyeccion("AAPL", start_date=start_date, end_date=end_date, window=50, plot=True)
print("Pendiente obtenida:", pendientes)

# Bandas de Bollinger
datos_bollinger = opt.bandas_bollinger(["AAPL", "MSFT"], start_date=start_date, end_date=end_date, window=20, plot=True)
```

### 3. Simulación y optimización de portafolios
```python
# Simular 5,000 portafolios aleatorios
weights, ret_esp, vol_esp, sharpe = opt.simular_portafolios(rendimientos_log, num_portafolios=5000)

# Encontrar el portafolio de Máximo Ratio de Sharpe
best_weights, ret_opt, vol_opt, max_sharpe = opt.encontrar_mejor_portafolio(
    weights, ret_esp, vol_esp, sharpe
)
opt.mostrar_resultados(tickers, best_weights, ret_opt, vol_opt, max_sharpe)

# Encontrar el portafolio de Mínima Varianza
min_w, min_ret, min_vol, min_s = opt.encontrar_minima_varianza(
    weights, ret_esp, vol_esp, sharpe
)
```

### 4. Medición de Riesgo (VaR y CVaR)
```python
import numpy as np

time_horizon = 100            # Horizonte a 100 días
inversion_inicial = 100_000.0 # $100,000 USD

# Rendimiento del portafolio combinado
ret_portafolio = rendimientos.dot(best_weights)

# 1. VaR y CVaR Histórico al 95% de confianza (alpha=5)
h_var = -opt.historical_var(ret_portafolio, alpha=5.0) * np.sqrt(time_horizon)
h_cvar = -opt.historical_cvar(ret_portafolio, alpha=5.0) * np.sqrt(time_horizon)

# 2. Simulación Monte Carlo
portfolio_sims, port_results = opt.monte_carlo_sim(
    mc_sims=1000,
    T=time_horizon,
    media_rendimiento=media_rend,
    peso=best_weights,
    initialPortfolio=inversion_inicial,
    covmatrix=covmatrix,
    plot=False,
)

# VaR y CVaR por Monte Carlo
mc_var_val = inversion_inicial - opt.mc_var(port_results, alpha=5.0)
mc_cvar_val = inversion_inicial - opt.mc_cvar(port_results, alpha=5.0)

# Desempeño estimado
p_ret, p_std = opt.desempeno(best_weights, media_rend, covmatrix, time=time_horizon)

# Resumen estructurado
opt.resumen_riesgo(
    inversion_inicial=inversion_inicial,
    initialPortfolio=inversion_inicial,
    hVaR=h_var,
    hCVaR=h_cvar,
    MCVaR=mc_var_val,
    MCCVaR=mc_cvar_val,
    pRet=p_ret,
)
```

### 5. Análisis de Sentimiento Financiero (NLP)
```python
from opt_pyFE import analizar_sentimiento

noticias = [
    "Apple beats quarterly revenue expectations driven by iPhone sales.",
    "Central bank unexpectedly raises interest rates sparking market sell-off."
]

resultados = analizar_sentimiento(noticias, forzar_binario=True)
print(resultados)
```

---

## Uso desde la Terminal (CLI)

El paquete incluye una interfaz de comandos `opt-pyfe`:
```bash
# Ver versión
opt-pyfe version

# Optimizar portafolio directamente en la terminal
opt-pyfe optimize AAPL MSFT NVDA GOOGL --start 2023-01-01 --end 2024-01-01 --sims 4000
```

---

## Tabla de Equivalencias y Retrocompatibilidad

Para garantizar que el código previo de los alumnos no se rompa, se mantienen alias automáticos entre los nombres clásicos y los nombres estándar (PEP 8):

| Función Estándar (PEP 8) | Alias Retrocompatible | Descripción |
| :--- | :--- | :--- |
| `get_data(...)` | `getdata(...)` | Descarga de datos y covarianzas |
| `bollinger_bands(...)` | `bandas_bollinger(...)` | Bandas de Bollinger |
| `simulate_portfolios(...)` | `simular_portafolios(...)`| Simulación Monte Carlo de pesos |
| `portfolio_performance(...)` | `desempeno(...)` | Rendimiento y riesgo proyectado |
| `historical_var(...)` | `historicalVar(...)` | VaR histórico |
| `historical_cvar(...)` | `historicalCVar(...)` | CVaR histórico |
| `monte_carlo_sim(...)` | `MonteCarlo(...)` | Simulación Monte Carlo multivariada |
| `mc_var(...)` | `mcVaR(...)` | VaR de Monte Carlo |
| `mc_cvar(...)` | `mcCVaR(...)` | CVaR de Monte Carlo |
| `resumen_riesgo(...)` | `resum(...)` | Resumen de métricas de riesgo |
| `analyze_sentiment(...)` | `analizar_sentimiento(...)`| Clasificación FinancialBERT |

---

## Ejecución de Pruebas Unitarias

Para ejecutar la batería completa de pruebas:
```bash
pytest -v
```

---

## Solución de Problemas Comunes (Troubleshooting)

### Límites de solicitudes en Yahoo Finance (`Too Many Requests`):
Si `yfinance` responde con errores de rate limiting o devuelve `DataFrame` vacío:
1. Actualiza `yfinance`: `pip install --upgrade yfinance`
2. Si realizas muchas peticiones repetidas, agrega pausas (`time.sleep`) o prueba utilizando una VPN o red distinta.
3. Asegúrate de pasar fechas con formato válido `YYYY-MM-DD`.

---

## Licencia

Distribuido bajo la Licencia **MIT**. Consulta el archivo `LICENSE` para más información.
