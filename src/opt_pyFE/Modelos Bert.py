# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:43:44 2025

@author: betoh
"""
import numpy as np
import matplotlib.pyplot as plt
# Importar la función `load_dataset` de la librería `datasets` de Hugging Face.
# Esta función facilita la carga de conjuntos de datos.
from datasets import load_dataset
# Importar la función `pipeline` de la librería `transformers` de Hugging Face.
# Esta función simplifica el uso de modelos preentrenados para tareas de NLP.
from transformers import pipeline
# Importa varias funciones de métricas de la librería scikit-learn (`sklearn`)
# para evaluar el rendimiento del modelo de clasificación.
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ----------------------------
# 1) Cargar datos y procesarlos
# ----------------------------
# Definir la ruta al archivo CSV que contiene los datos. 
#La `r` antes de la cadena indica una 'raw string'.
csv_path = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\Phyton\webscrapping\Analisis de sentimiento\2 datasets de finanzas.csv"

# Utilizar `load_dataset` para cargar los datos desde el CSV especificado.
data = load_dataset(
    "csv",  # Indica que el formato del archivo es CSV.
    data_files=csv_path,  # Proporciona la ruta del archivo.
    encoding="latin-1",  # Especifica la codificación de caracteres del archivo CSV
    #(común para archivos con caracteres españoles no guardados en UTF-8).
    column_names=["label", "text"]  # Asignar nombres a las columnas del CSV:
        #                       'label' para la etiqueta y 'text' para el texto.
)["train"].class_encode_column("label")  # Selecciona la partición 'train' y 
    #convierte la columna 'label' a un formato numérico interno, guardando el mapeo.

# Se utiliza todo el dataset para la evaluación.
# Extraer la columna de texto del dataset cargado y la asigna a `test_txt`.
# Estos serán los datos de entrada para el modelo.
test_txt = data["text"]
# Extraer la columna de etiquetas (ya codificadas numéricamente) del dataset y 
#la asigna a `y_true`. Estas son las etiquetas verdaderas.
y_true   = data["label"]
# Obtiene la lista de nombres de las clases originales
# a partir de las características del dataset.
classes  = data.features["label"].names
# Calcular el número total de clases distintas.
num_classes = len(classes)
# Crear un diccionario que mapea cada nombre de clase (string)
# a su índice numérico correspondiente (ID).
label2id = {n: i for i, n in enumerate(classes)}

# ----------------------------
# 2) Configurar pipelines (modelos)
# ----------------------------
# Definir una lista de identificadores de modelos preentrenados de Hugging Face
# Actualmente, solo está activo 'ahmedrachid/FinancialBERT-Sentiment-Analysis'.
# Los otros están comentados debido a su bajo accuracy 
models = [
    #"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
     "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    # "soleimanian/financial-roberta-large-sentiment"
]

# Crea una lista de `pipeline` de Hugging Face,
# uno para cada modelo en la lista `models`.
pipes = [
    pipeline("text-classification",  # Inicializar un pipeline para la tarea de
             #'clasificación de texto'.
             model=m,  # Utiliza el modelo `m` de la lista `models`.
             device=-1,  # Especifica que el modelo debe ejecutarse en la CPU.
             return_all_scores=True,  # Indica al pipeline que devuelva las puntuaciones para todas las clases.
             truncation=True,  # Habilita la truncación de los textos de entrada.
             padding=True,  # Habilita el padding de los textos de entrada.
             batch_size=32)  # Define el tamaño del lote para el procesamiento.
    for m in models  # Iterar sobre cada modelo en la lista `models`.
]

# ----------------------------
# 3) Inferencia: ensamblaje de predicciones
# ----------------------------
# Se suman las probabilidades para cada clase.
# Inicializar un array de NumPy con ceros para acumular las puntuaciones 
#de los modelos para cada texto y cada clase.
logits_sum = np.zeros((len(test_txt), num_classes))
# Iterar sobre cada pipeline (modelo) configurado.
for pl in pipes:
    # Pasa todos los textos de `test_txt` al pipeline actual (`pl`) para 
    # obtener las predicciones.
    results = pl(test_txt)
    # Itera sobre los resultados, donde `i` es el índice del texto y `res` 
    #es la lista de diccionarios de puntuaciones para ese texto.
    for i, res in enumerate(results):
        # Itera sobre cada diccionario de puntuación (`d`) dentro de `res`.
        for d in res:
            # Acumula las puntuaciones: para el texto `i`, 
            #localiza la columna de la clase `d["label"]` (usando `label2id`) y
            #suma la puntuación `d["score"]`.
            logits_sum[i, label2id[d["label"]]] += d["score"]

# Para cada texto, encuentra el índice de la clase con la suma de \
# puntuaciones más alta. Estas son las predicciones finales del ensamblaje.
preds = logits_sum.argmax(axis=1)

# Métricas globales.
# Calcula e imprime la métrica de 'accuracy' (exactitud) del ensamblaje.
print("Ensemble Accuracy :", accuracy_score(y_true, preds))
# Calcula e imprime la métrica F1-score promediada como 'macro' del ensamblaje.
print("Ensemble F1‑macro :", f1_score(y_true, preds, average="macro"))
 
# Genera un reporte de clasificación que incluye precisión, 
# recall y F1-score por clase.
# `target_names` usa los nombres de las clases para el reporte.
# `output_dict=True` devuelve un diccionario.
report_dict = classification_report(y_true, preds, target_names=classes, output_dict=True)

# ----------------------------
# 4) Gráficas de Evaluación
# ----------------------------

# 4.1. Matriz de Confusión.
# Calcula la matriz de confusión usando las etiquetas verdaderas y las predicciones.
# `labels` asegura que las filas/columnas sigan el orden de 0 a `num_classes-1`.
cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
# Crea una nueva figura para el gráfico con un tamaño específico.
plt.figure(figsize=(6, 5))
# Muestra la matriz de confusión como una imagen, usando una escala de colores azules.
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# Añade título al gráfico.
plt.title("Matriz de Confusión")
# Añade una barra de color para interpretar los valores de la matriz.
plt.colorbar()
# Establece las etiquetas del eje X (predichas) con los nombres de las clases.
plt.xticks(np.arange(num_classes), classes)
# Establece las etiquetas del eje Y (verdaderas) con los nombres de las clases.
plt.yticks(np.arange(num_classes), classes)
# Calcula un umbral para decidir el color del texto dentro de las celdas 
#(blanco sobre oscuro, negro sobre claro).
threshold = cm.max() / 2.0
# Itera sobre cada celda de la matriz de confusión.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # Añade el valor numérico de la celda en su centro.
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > threshold else "black")
# Añade etiqueta al eje Y.
plt.ylabel("Etiqueta verdadera")
# Añade etiqueta al eje X.
plt.xlabel("Etiqueta predicha")
# Ajusta automáticamente los parámetros del subplot para un diseño ajustado.
plt.tight_layout()
# Muestra el gráfico.
plt.show()

# 4.2. Gráfica de Precision, Recall y F1‑Score por clase.
# Obtiene las etiquetas de clase que están presentes en el `report_dict`.
class_labels = [lbl for lbl in classes if lbl in report_dict]
# Extrae los valores de precisión para cada clase del `report_dict`.
precision_vals = [report_dict[lbl]['precision'] for lbl in class_labels]
# Extrae los valores de recall para cada clase del `report_dict`.
recall_vals    = [report_dict[lbl]['recall'] for lbl in class_labels]
# Extrae los valores de F1-score para cada clase del `report_dict`.
f1_vals        = [report_dict[lbl]['f1-score'] for lbl in class_labels]

# Crea un array de posiciones para las barras en el eje X.
x = np.arange(len(class_labels))
# Define el ancho de cada barra en el gráfico de barras agrupado.
bar_width = 0.25

# Crea una nueva figura para el gráfico.
plt.figure(figsize=(8, 5))
# Crea las barras para la precisión.
bars1 = plt.bar(x, precision_vals, bar_width, label='Precision')
# Crea las barras para el recall, desplazadas para agruparlas con las de precisión.
bars2 = plt.bar(x + bar_width, recall_vals, bar_width, label='Recall')
# Crea las barras para el F1-score, desplazadas para agruparlas.
bars3 = plt.bar(x + 2 * bar_width, f1_vals, bar_width, label='F1-Score')

# Establece la etiqueta del eje X.
plt.xlabel('Clases')
# Establece la etiqueta del eje Y.
plt.ylabel('Puntuación')
# Establece el título del gráfico.
plt.title('Métricas por Clase')
# Establece las etiquetas de las marcas del eje X con los nombres de las clases, centradas en el grupo de barras.
plt.xticks(x + bar_width, class_labels)
# Muestra la leyenda del gráfico.
plt.legend()

# Se agregan data labels para cada métrica.
# Itera sobre las barras de precisión para añadir etiquetas de datos.
for i, bar in enumerate(bars1):
    height = bar.get_height()
    # Añade el valor numérico de la precisión encima de la barra.
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{precision_vals[i]:.2f}",
             ha='center', va='bottom')
# Itera sobre las barras de recall para añadir etiquetas de datos.
for i, bar in enumerate(bars2):
    height = bar.get_height()
    # Añade el valor numérico del recall encima de la barra.
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{recall_vals[i]:.2f}",
             ha='center', va='bottom')
# Itera sobre las barras de F1-score para añadir etiquetas de datos.
for i, bar in enumerate(bars3):
    height = bar.get_height()
    # Añade el valor numérico del F1-score encima de la barra.
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{f1_vals[i]:.2f}",
             ha='center', va='bottom')

# Ajusta automáticamente los parámetros del subplot.
plt.tight_layout()
# Muestra el gráfico.
plt.show()

# 4.3. Gráfica de F2‑score por clase.
# Fórmula: F2 = (5 * precision * recall) / (4 * precision + recall) (si el denominador no es cero).
# El F2-score da más peso al recall que a la precisión.
# Inicializa una lista para almacenar los valores de F2-score.
f2_vals = []
# Itera sobre cada nombre de clase.
for c in classes:
    # Obtiene la precisión para la clase `c` del `report_dict`.
    prec = report_dict[c]["precision"]
    # Obtiene el recall para la clase `c` del `report_dict`.
    rec = report_dict[c]["recall"]
    # Calcula el F2-score. Maneja la división por cero asignando 0 si el denominador es cero.
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) else 0
    # Añade el F2-score calculado a la lista.
    f2_vals.append(f2)

# Crea una nueva figura para el gráfico.
plt.figure(figsize=(7, 4))
# Crea un gráfico de barras para los F2-scores por clase.
bar_f2 = plt.bar(classes, f2_vals, color="purple")
# Establece los límites del eje Y entre 0 y 1.
plt.ylim(0, 1)
# Establece el título del gráfico.
plt.title("F2‑score por clase")
# Establece la etiqueta del eje Y.
plt.ylabel("F2‑score")
# Itera sobre las barras del F2-score para añadir etiquetas de datos.
for bar, val in zip(bar_f2, f2_vals):
    # Añade el valor numérico del F2-score encima de la barra.
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha="center", va="bottom")
# Ajusta automáticamente los parámetros del subplot.
plt.tight_layout()
# Muestra el gráfico.
plt.show()