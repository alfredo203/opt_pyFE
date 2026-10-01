"""Tutorial completo de uso de trading_unam (Trading UNAM).

Este script demuestra el flujo de trabajo financiero completo:
1. Descarga y cálculo de rendimientos.
2. Analítica técnica (regresión lineal y bandas de Bollinger).
3. Simulación y optimización de portafolios (Ratio de Sharpe y Mínima Varianza).
4. Medición de riesgo (VaR y CVaR histórico y Monte Carlo).
5. Análisis de sentimiento en noticias financieras (opcional con Transformers).
"""

import datetime as dt
import numpy as np
import trading_unam as tu


def main():
    print("=" * 60)
    print("        TUTORIAL: FLUJO COMPLETO CON Trading UNAM")
    print("=" * 60)

    # 1. Definición de parámetros
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    end_date = dt.date.today().strftime("%Y-%m-%d")
    start_date = (dt.date.today() - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    inversion_inicial = 100_000.0  # $100,000 MXN / USD
    time_horizon = 100             # 100 días

    print(f"\n[1] Activos: {tickers}")
    print(f"    Periodo: {start_date} al {end_date}")

    # 2. Descarga de datos y rendimientos
    print("\n[2] Descargando datos y calculando rendimientos...")
    df_precios = tu.descargar_datos(tickers, start_date=start_date, end_date=end_date)
    print("    Precios descargados:\n", df_precios.tail(3))

    rendimiento, media_rendimiento, covmatrix = tu.getdata(tickers, start=start_date, end=end_date)
    log_returns = tu.calcular_rendimientos_log(df_precios)
    print("\n    Matriz de covarianza diaria:\n", covmatrix)

    # 3. Analítica técnica
    print("\n[3] Analítica técnica (Media Móvil, Regresión y Bollinger)...")
    # Regresión lineal para el primer ticker (sin bloquear con show continuo)
    pendientes = tu.proyeccion(tickers[0], start_date=start_date, end_date=end_date, window=50, plot=False)
    print(f"    Pendiente de regresión lineal para {tickers[0]}: {pendientes[0][1]:.4f}")

    # Bandas de Bollinger para el activo
    bollinger = tu.bandas_bollinger(tickers[0], start_date=start_date, end_date=end_date, window=20, plot=False)
    print(f"    Últimos valores de Bollinger para {tickers[0]}:\n", bollinger[tickers[0]][["Close", "MA20", "UpperBand", "LowerBand"]].tail(3))

    # 4. Simulación y optimización de portafolios (Markowitz)
    print("\n[4] Simulación de portafolios (5,000 iteraciones)...")
    num_portafolios = 5000
    weights, ret_esp, vol_esp, sharpe = tu.simular_portafolios(log_returns, num_portafolios=num_portafolios)

    best_weights, best_ret, best_vol, max_sharpe = tu.encontrar_mejor_portafolio(
        weights, ret_esp, vol_esp, sharpe
    )
    tu.mostrar_resultados(tickers, best_weights, best_ret, best_vol, max_sharpe)

    min_w, min_r, min_v, min_s = tu.encontrar_minima_varianza(
        weights, ret_esp, vol_esp, sharpe
    )
    print(f"\n    Portafolio de Mínima Varianza -> Volatilidad: {min_v:.4f} | Retorno: {min_r:.4f}")

    # 5. Desempeño y Medición de Riesgo (VaR y CVaR)
    print("\n[5] Gestión de Riesgo (VaR y CVaR al 95% de confianza)...")
    p_ret, p_std = tu.desempeno(best_weights, media_rendimiento, covmatrix, time=time_horizon)

    # Rendimiento histórico del portafolio combinado
    rendimiento_portafolio = rendimiento.dot(best_weights)

    # VaR y CVaR Histórico
    h_var = -tu.historical_var(rendimiento_portafolio, alpha=5.0) * np.sqrt(time_horizon)
    h_cvar = -tu.historical_cvar(rendimiento_portafolio, alpha=5.0) * np.sqrt(time_horizon)

    # Simulación Monte Carlo
    print("    Ejecutando Simulación Monte Carlo multivariada (1,000 caminos)...")
    sims, port_results = tu.monte_carlo_sim(
        mc_sims=1000,
        T=time_horizon,
        media_rendimiento=media_rendimiento,
        peso=best_weights,
        initialPortfolio=inversion_inicial,
        covmatrix=covmatrix,
        plot=False,
    )

    mc_var_val = inversion_inicial - tu.mc_var(port_results, alpha=5.0)
    mc_cvar_val = inversion_inicial - tu.mc_cvar(port_results, alpha=5.0)

    # Resumen final de riesgo
    tu.resumen_riesgo(
        inversion_inicial=inversion_inicial,
        initialPortfolio=inversion_inicial,
        hVaR=h_var,
        hCVaR=h_cvar,
        MCVaR=mc_var_val,
        MCCVaR=mc_cvar_val,
        pRet=p_ret,
    )

    print("\n[OK] ¡Tutorial completado con éxito!")


if __name__ == "__main__":
    main()
