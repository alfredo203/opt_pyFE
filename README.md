# opt_pyFE

[![PyPI version](https://img.shields.io/pypi/v/opt_pyFE.svg)](https://pypi.org/project/opt_pyFE/)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

### Instalación vía PyPI (Recomendado):
Una vez publicado en PyPI, instala directamente con:
```bash
pip install opt_pyFE
```
*(o alternativamente `pip install opt-pyfe`)*

Con soporte para procesamiento de lenguaje natural (NLP / FinancialBERT):
```bash
pip install "opt_pyFE[nlp]"
```

### Instalación local desde el código fuente (Desarrollo):
```bash
git clone https://github.com/alfredo203/opt_pyFE.git
cd opt_pyFE
pip install -e ".[dev]"
```

---

## Estructura del Proyecto

```text
opt_pyFE/
├── .github/
│   └── workflows/
│       └── publish.yml        # Publicación automatizada a PyPI vía GitHub Actions
├── pyproject.toml             # Configuración moderna de empaquetado (PEP 518/621)
├── MANIFEST.in                # Manifiesto de archivos fuente para distribución
├── README.md                  # Documentación principal
├── src/
│   └── opt_pyFE/
│       ├── __init__.py        # API pública limpia y controlada (__all__)
│       ├── data.py            # Descarga de datos y rendimientos
│       ├── analytics.py       # Medias móviles, regresión y Bollinger
│       ├── portfolio.py       # Simulación de portafolios y Sharpe
│       ├── risk.py            # VaR, CVaR y simulación Monte Carlo
│       ├── sentiment.py       # Análisis de sentimiento con FinancialBERT
│       ├── cli.py             # Herramienta de línea de comandos (Typer/Rich)
│       └── opt_pyFE.py        # Módulo unificado para retrocompatibilidad
├── examples/
│   ├── tutorial_opt_pyFE.py   # Tutorial completo ejecutable de inicio a fin
│   └── bert_sentiment_demo.py # Demo interactiva de análisis de sentimiento
└── tests/                     # Suite completa de pruebas unitarias (pytest)
    ├── test_data.py
    ├── test_analytics.py
    ├── test_portfolio.py
    ├── test_risk.py
    ├── test_imports.py
    └── test_opt_pyFE.py
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
# Proyección con regresión lineal y media móvil
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

## Publicación en PyPI (Guía Paso a Paso)

El paquete se encuentra empaquetado bajo los estándares modernos de Python (**PEP 517 / PEP 518 / PEP 621** con `pyproject.toml`).

### Paso 1: Instalar herramientas de compilación
Asegúrate de contar con `build` y `twine`:
```bash
pip install --upgrade build twine
```

### Paso 2: Limpiar artefactos previos y compilar la distribución
Genera el paquete de código fuente (`.tar.gz`) y el archivo binario (`.whl`):
```bash
python -m build
```
Los archivos compilados quedarán almacenados en la carpeta `dist/`.

### Paso 3: Validar la integridad del paquete
Verifica que los metadatos y el README cumplan las especificaciones de PyPI:
```bash
twine check dist/*
```
*(Debe reportar `PASSED` en todos los archivos).*

### Paso 4 (Opcional pero recomendado): Probar subida en TestPyPI
Para verificar que el registro sea exitoso sin afectar la versión de producción:
1. Crea una cuenta en [TestPyPI](https://test.pypi.org/) y genera un **API Token**.
2. Sube la distribución a TestPyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```
   * En usuario ingresa: `__token__`
   * En contraseña ingresa el token con prefijo: `pypi-...`
3. Comprueba la instalación desde TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --no-deps opt_pyFE
   ```

### Paso 5: Publicación oficial en PyPI (Producción)
1. Crea una cuenta en [PyPI oficial](https://pypi.org/) y activa autenticación de dos factores (2FA).
2. Genera un **API Token** en *Account Settings* > *API Tokens*.
3. Sube los archivos a PyPI:
   ```bash
   twine upload dist/*
   ```
   * En usuario ingresa: `__token__`
   * En contraseña pega tu token: `pypi-...`

¡Listo! A partir de ese momento, cualquier persona en el mundo podrá instalar la librería con `pip install opt_pyFE`.

### Paso 6: Publicación automatizada mediante GitHub Actions (Opcional)
El repositorio ya incluye el workflow [`.github/workflows/publish.yml`](.github/workflows/publish.yml). Para publicar automáticamente:
1. Ve a la configuración de tu repositorio en PyPI y habilita **Trusted Publishing** vinculando `alfredo203/opt_pyFE`.
2. O bien agrega el secreto `PYPI_API_TOKEN` en tu repositorio de GitHub (*Settings > Secrets and variables > Actions*).
3. Cada vez que crees un **Release** en GitHub, el paquete se compilará, correrá los tests y se publicará en PyPI de forma 100% desatendida.

---

## Tabla de Equivalencias y Retrocompatibilidad

Para garantizar que el código previo no se rompa, se mantienen alias automáticos entre los nombres clásicos y los nombres estándar (PEP 8):

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
