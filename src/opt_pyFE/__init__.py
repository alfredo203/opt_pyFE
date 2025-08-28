"""Top-level package for opt_pyFE."""

__author__ = """Equipo Python FE"""
__email__ = 'afredo.olguin@economia.unam.mx'
__version__ = '0.1.1'

# Archivos locales
from .opt_pyFE import *
# from defmodelmach import *
# from Empaquetado_bandas_bollinger_analiticos import *


# Dependencias externas (pip install)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt