# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 18:15:03 2025

@author: guadalupe
"""

"""
Script para descargar datos de Yahoo Finance y guardarlos en MySQL
"""

import yfinance as yf
import mysql.connector

# Configuración de la conexión a una base de datos MySQL desde Python
MYSQL_USER = "root"   # Usuario administrador por defecto en MySQL.     
MYSQL_PASSWORD = "1234"       # Cambia si tu root tiene contraseña
MYSQL_HOST = "localhost"   #Indica que la base de datos está en el mismo equipo que la aplicación.
MYSQL_PORT = 3306         #Puerto de conexión que usa MySQL por defecto.
DB_NAME = "finanzas"

TICKERS = tickers = [
    "AA","AAL","AAPL","AMM.TO","ABBV","ABNB","ACTINVRB.MX","AC","AFRM",
    "AGNC","ALFAA.MX","ALPEKA.MX","ALSEA.MX","AMAT","AMD","AMX","AMZN",
    "APA","ASURB.MX","ATER","ATOS","AIY.DE","AVGO","AXP","BABA","BAC",
    "BA","BBAJIOO.MX","BIMBOA.MX","BMY","BNGO","CAT","CCL",
    "CEMEXCPO.MX","CHDRAUIB.MX","CLF","COST","CRM","CSCO",
    "CUERVO.MX","CVS","CVX","C","DAL","DIS","DVN","ELEKTRA.MX","ETSY",
    "FANG","FCX","FDX","FEMSAUBD.MX","FIBRAMQ12.MX","FIBRAPL14.MX",
    "FSLR","FUBO","FUNO11.MX","F","GAPB.MX","GCARSOA1.MX","GCC",
    "GENTERA.MX","GE","GFINBURO.MX","GFNORTEO.MX","GILD","GMEXICOB.MX",
    "GME","GM","GOLD","GOOGL","GRUMAB.MX","HD","INTC","JNJ","JPM",
    "KIMBERA.MX","KOFUBL.MX","KO","LABB.MX",
    "LASITEB-1.MX","LCID","LIVEPOLC-1.MX","LLY","LUV","LVS","LYFT","MARA",
    "MA","MCD","MEGACPO.MX","MELIN.MX","META","MFRISCOA-1.MX","MGM",
    "MRK","MRNA","MRO","MSFT","MU","NCLHN.MX","NFLX","NKE","NKLA","NUN.MX",
    "NVAX","NVDA","OMAB.MX","ORBIA.MX","ORCL","OXY1.MX","PARA","PBRN.MX","PE&OLES.MX",
    "PEP","PFE","PG","PINFRA.MX","PINS","PLTR","PYPL","QCOM","Q.MX","RCL",
    "RIOT","RIVN","ROKU","RA.MX","SBUX","SHOP","SITES1A-1.MX","SKLZ",
    "SOFI","SPCE","SQ","TALN.MX","TERRA13.MX","TGT","TLEVISACPO.MX","TMO",
    "TSLA","TSMN.MX","TWLO","TX","T","UAL","UBER","UNH","UPST","VESTA.MX",
    "VOLARA.MX","VZ","V","WALMEX.MX","WFC","WMT","WYNN","XOM","X","ZM"
]  # Lista de acciones
START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

# ------ Crear una base de datos en MySQL si no existe aún-----------------------
try:
    conexion = mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        port=MYSQL_PORT
    )
    cursor = conexion.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    print(f"Base de datos '{DB_NAME}' lista.")
except mysql.connector.Error as err:
    print(f"Error al crear la base de datos: {err}")
    exit()
finally:
    cursor.close()
    conexion.close()

# ------------------ CONECTAR A LA BASE DE DATOS ------------------
db_config = {
    "host": MYSQL_HOST,
    "user": MYSQL_USER,
    "password": MYSQL_PASSWORD,
    "database": DB_NAME,
    "port": MYSQL_PORT
} #Agrupa todos los parámetros necesarios para conectarte a la base de datos.

try:
    conexion = mysql.connector.connect(**db_config)
    cursor = conexion.cursor()
except mysql.connector.Error as err:
    print(f"Error al conectar a la base de datos: {err}")
    exit() #Establece la conexión con los parámetros definidos.

# ------------------ CREAR TABLA ------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS acciones (
    fecha DATE,
    ticker VARCHAR(10),
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    PRIMARY KEY (fecha, ticker)
)
""")
print("Tabla 'acciones' lista.")

# ------------------ DESCARGAR DATOS Y GUARDAR ------------------
for ticker in TICKERS:
    print(f"\nDescargando datos de {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    
    if df.empty:
        print(f"No se encontraron datos para {ticker}.")
        continue  # Salta al siguiente ticker
    
    for fecha, row in df.iterrows():
        try:
            cursor.execute("""
            INSERT INTO acciones (fecha, ticker, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open=VALUES(open),
                high=VALUES(high),
                low=VALUES(low),
                close=VALUES(close),
                volume=VALUES(volume)
            """, (
                fecha.date(),        # índice de fecha
                ticker,              # ticker
                float(row["Open"]),  # convertir a float
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"])   # convertir volumen a entero
            ))
        except mysql.connector.Error as err:
            print(f"Error al insertar datos: {err}")

conexion.commit()
print("\nDatos guardados correctamente.")

# ------------------ RESUMEN AUTOMÁTICO ------------------
cursor.execute("SELECT COUNT(*) FROM acciones")
total = cursor.fetchone()[0]
print(f"\nTotal de registros en la tabla: {total}")

cursor.execute("SELECT ticker, COUNT(*), MIN(fecha), MAX(fecha) FROM acciones GROUP BY ticker")
print("\nResumen por ticker:")
for fila in cursor.fetchall():
    print(f"Ticker: {fila[0]} - Registros: {fila[1]} - Fechas: {fila[2]} a {fila[3]}")

cursor.close()
conexion.close()