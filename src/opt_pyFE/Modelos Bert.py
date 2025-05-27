# -*- coding: utf-8 -*-
"""
Crear el script el Dom Abr  6 14:43:44 2025.
Modificar para clasificación interactiva y almacenar en DataFrame.

@author: betoh
"""
import pandas as pd
from transformers import pipeline

# ----------------------------
# 1) Configurar el pipeline (modelo)
# ----------------------------
model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
sentiment_pipeline = pipeline(
    "text-classification",
    model=model_name,
    device=-1, # Indicar el uso de CPU
    return_all_scores=True,
    truncation=True,
    padding=True # 
)

print(f"Pipeline configurado con el modelo: {model_name}")

# Definir los nombres de las etiquetas para lograr consistencia.
# El modelo FinancialBERT-Sentiment-Analysis utilizará:
positive_label_str = 'positive'
negative_label_str = 'negative'
neutral_label_str = 'neutral' # El modelo poder producir esta etiqueta.

# --------------------------------------------------------------------------
# Crear un DataFrame para almacenar los resultados.
# --------------------------------------------------------------------------
# Definir las columnas:
results_df = pd.DataFrame(columns=[
    "Noticia",
    "Clasificacion_Forzada",
    "Score_Positivo",
    "Score_Negativo",
    "Score_Neutral"
])

# --------------------------------------------------------------------------
# Iniciar la sección de Predicción Interactiva (con elección forzada P/N).
# --------------------------------------------------------------------------
print("\n\n PREDICCIÓN INTERACTIVA DE SENTIMIENTO (FORZADO POSITIVO/NEGATIVO)")
print("Escribir un titular de noticia (o 'exit' para salir):")

while True:
    user_input = input("> ")
    if user_input.lower() == 'exit':
        print("Salir del modo interactivo.")
        break
    if not user_input.strip():
        print("Por favor, ingresar algún texto.")
        continue

    # El pipeline va a devolver una lista de listas de diccionarios.
    # Para un solo input, ser una lista con UN elemento, que es la lista de scores.
    prediction_scores_list = sentiment_pipeline(user_input)[0]

    # Extraer los scores originales del modelo.
    raw_positive_score = 0.0
    raw_negative_score = 0.0
    raw_neutral_score = 0.0

    for item in prediction_scores_list:
        label = item['label'].lower() 
        score = item['score']
        if label == positive_label_str:
            raw_positive_score = score
        elif label == negative_label_str:
            raw_negative_score = score
        elif label == neutral_label_str:
            raw_neutral_score = score
        # Poder añadir más 'elif' si el modelo tuviera otras etiquetas.

    # Implementar la lógica para forzar la clasificación a Positivo o Negativo.
    sum_positive_negative = raw_positive_score + raw_negative_score
    forced_classification_str = ""
    # Considerar como opcional la confianza relativa entre P y N.

    if sum_positive_negative == 0:
        # Manejar el caso extremo: el modelo poder dar 0 a ambas clases de interés (ej. muy neutral o etiqueta no reconocida).
        # Forzar una elección, por ejemplo, tomar negativo por defecto.
        # O poder decidir basado en el score neutral si es muy alto, pero la petición es forzar P/N.
        forced_classification_str = negative_label_str.capitalize() + " (scores P/N originales eran 0)"
        # O usar 0.5 si se va a considerar un empate.
    else:
        # Normalizar los scores de las dos clases de interés (positivo y negativo).
        # normalized_positive = raw_positive_score / sum_positive_negative
        # normalized_negative = raw_negative_score / sum_positive_negative

        # Decidir basado en cuál de los dos (positivo o negativo) tener mayor score original.
        # Esto equivaler a ver si normalized_positive > 0.5 (o >= 0.5 para decidir un desempate).
        if raw_positive_score >= raw_negative_score:
            forced_classification_str = positive_label_str.capitalize()
            # relative_confidence_pn = normalized_positive
        else:
            forced_classification_str = negative_label_str.capitalize()
            # relative_confidence_pn = normalized_negative

    # Imprimir los resultados de la predicción actual.
    print(f"\nNoticia: \"{user_input}\"")
    print(f"Clasificación Forzada (P/N): {forced_classification_str}")
    print("  Scores originales del modelo:")
    print(f"    - Positivo: {raw_positive_score:.4f}")
    print(f"    - Negativo: {raw_negative_score:.4f}")
    print(f"    - Neutral:  {raw_neutral_score:.4f}")
    print("-" * 40)

    # Añadir la información al DataFrame.
    new_row = pd.DataFrame([{
        "Noticia": user_input,
        "Clasificacion_Forzada": forced_classification_str.split(" (")[0], # Extraer solo la etiqueta sin el comentario adicional.
        "Score_Positivo": raw_positive_score,
        "Score_Negativo": raw_negative_score,
        "Score_Neutral": raw_neutral_score
    }])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Al salir del bucle, el DataFrame 'results_df' deberá contener todos los datos.
# Poder imprimirlo aquí si se desear, o guardarlo, etc.
print("\n--- DataFrame con todos los resultados acumulados ---")
print(results_df)

# Ilustrar con un ejemplo cómo guardar el DataFrame a un archivo CSV al final:
# results_df.to_csv("historial_sentimientos.csv", index=False, encoding='utf-8-sig')
# print("\nResultados guardados en 'historial_sentimientos.csv'")

print("\n--- Fin del Script ---")

# Verificar si results_df existir y no estar vacío.
if 'results_df' in locals() and isinstance(results_df, pd.DataFrame) and not results_df.empty:
    try:
        # Definir el nombre del archivo Excel de salida.
        excel_file_name = "analisis_sentimiento_noticias.xlsx"

        # Exportar el DataFrame a un archivo Excel.
        # El parámetro index=False evitar que se escriba el índice del DataFrame como una columna en Excel.
        # El parámetro sheet_name permitir nombrar la hoja dentro del archivo Excel.
      
      #!!! Descomentar para poder guardar el excel con las noticias evaluadas" 
      #  results_df.to_excel(excel_file_name, index=False, sheet_name="ResultadosSentimiento")

        print(f"\nEl DataFrame haber sido exportado exitosamente a '{excel_file_name}'")

    except ImportError:
        print("\nError: La librería 'openpyxl' ser necesaria para exportar a Excel (.xlsx).")
        print("Por favor, instalarla ejecutando: pip install openpyxl")
    except Exception as e:
        print(f"\nOcurrir un error al intentar exportar el DataFrame a Excel: {e}")
else:
    print("\nLa variable 'results_df' no se encontrar, estar vacía o no ser un DataFrame.")
    print("Asegurar que el script principal se haya ejecutado correctamente y haya poblado 'results_df'.")