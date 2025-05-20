# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:43:44 2025 

@author: betoh
"""
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, concatenate_datasets, Dataset, ClassLabel
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ----------------------------
# 1) Cargar datos originales
# ----------------------------
csv_path = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\Phyton\webscrapping\Analisis de sentimiento\2 datasets de finanzas.csv"
data_original = load_dataset(
    "csv",
    data_files=csv_path,
    encoding="latin-1",
    column_names=["label", "text"]
)["train"].class_encode_column("label")

print(f"Tamaño del dataset original: {len(data_original)}")
original_class_names = data_original.features["label"].names
print("Distribución original de clases:")
for class_name in original_class_names:
    class_id = data_original.features['label'].str2int(class_name)
    count = len(data_original.filter(lambda x: x['label'] == class_id))
    print(f"- {class_name}: {count}")

# ----------------------------
# 2) Configurar pipeline (modelo)
# ----------------------------
model_name = "ahmedrachid/FinancialBERT-Sentiment-Analysis"
sentiment_pipeline = pipeline(
    "text-classification",
    model=model_name,
    device=-1, # CPU
    return_all_scores=True,
    truncation=True,
    padding=True,
    batch_size=32
)

print(f"\nPipeline configurado con el modelo: {model_name}")

# ---------------------------------------------------------------------------------
# Funciones Auxiliares para Inferencia y Visualización (para reusar)
# ---------------------------------------------------------------------------------

def create_binary_hf_dataset(texts_list, string_labels_list, class_order_list):
    """
    Crea un Dataset de Hugging Face con etiquetas binarias codificadas.
    class_order_list: Define el mapeo, ej: ['negative', 'positive'] -> 0, 1
    """
    temp_dataset = Dataset.from_dict({"text": texts_list, 
                                      "label_str": string_labels_list})
    actual_present_labels = sorted(list(set(string_labels_list)))
    if not all(lbl in class_order_list for lbl in actual_present_labels):
        print(f"Advertencia: Las etiquetas presentes {actual_present_labels} no coinciden completamente con class_order_list {class_order_list}. Usando etiquetas presentes.")
        class_order_list_effective = actual_present_labels
    else:
        class_order_list_effective = [lbl for 
                                      lbl in class_order_list if 
                                      lbl in actual_present_labels]

    if not class_order_list_effective:
        print("Error: No hay etiquetas para crear el ClassLabel.")
        return Dataset.from_dict({"text": [], 
                                  "label": []}).cast_column('label',
                                                            ClassLabel(names=[]))

    feature_class_label = ClassLabel(names=class_order_list_effective)
    final_dataset = temp_dataset.cast_column('label_str', feature_class_label)
    final_dataset = final_dataset.rename_column('label_str', 'label')
    return final_dataset.shuffle(seed=42)


def perform_binary_inference_forced_choice(dataset_binary, 
                                           pipeline_model, 
                                           binary_classes_names_ordered):
    """
    Realiza inferencia en un dataset binario FORZANDO una elección entre las 
    dos clases binarias, normalizando sus scores.
    binary_classes_names_ordered: Lista de nombres de clase en el orden deseado
    para 0, 1 (ej: ['negative', 'positive'])
    """
    test_txt = dataset_binary["text"]
    # Ya son 0, 1 según binary_classes_names_ordered
    y_true = dataset_binary["label"] 
    
    # binary_classes_names_ordered[0] será la clase mapeada a 0 (ej.'negative')
    # binary_classes_names_ordered[1] será la clase mapeada a 1 (ej.'positive')

    preds = [] # Lista para guardar las predicciones finales (0 o 1)
    
  # Esto devuelve una lista de listas de diccionarios
    results_batch = pipeline_model(test_txt) 
    for i, res_item_list in enumerate(results_batch):
        # res_item_list es una lista de diccionarios,
        # ej: [{'label': 'positive', 'score': 0.9}, {'label': 'negative', ...}]
        
        score_class0_original = 0.0 # Score para binary_classes_names_ordered[0]
        score_class1_original = 0.0 # Score para binary_classes_names_ordered[1]

        for score_dict in res_item_list:
            model_output_label_str = score_dict["label"].lower() # e.g., 'positive', 'negative', 'neutral'
            
            if model_output_label_str == binary_classes_names_ordered[0].lower():
                score_class0_original = score_dict["score"]
            elif model_output_label_str == binary_classes_names_ordered[1].lower():
                score_class1_original = score_dict["score"]
            # Los scores de 'neutral' (o cualquier otra clase no en binary_classes_names_ordered) se ignoran aquí

        sum_scores_class0_class1 = score_class0_original + score_class1_original

        if sum_scores_class0_class1 == 0:
            # Caso extremo: el modelo dio 0 a ambas clases de interés.
            # Forzamos una elección, por ejemplo, la clase 0 (negative por defecto en nuestro orden).
            # O podríamos alternar, o aleatorio. Para consistencia, elegimos la clase 0.
            predicted_label_idx = 0 
        else:
            # Normalizar los scores de las dos clases de interés
            normalized_score_class0 = score_class0_original / sum_scores_class0_class1
            # normalized_score_class1 = score_class1_original / sum_scores_class0_class1 # No es necesario calcularla explícitamente para la decisión

            if normalized_score_class0 > 0.5: # Si la clase 0 tiene más del 50% de la probabilidad relativa
                predicted_label_idx = 0
            elif normalized_score_class0 < 0.5: # Si la clase 1 tiene más del 50%
                predicted_label_idx = 1
            else: # Empate exacto (0.5 vs 0.5), elegimos la clase 0 por defecto (o la 1, sé consistente)
                predicted_label_idx = 0 
        
        preds.append(predicted_label_idx)

    return y_true, np.array(preds)


def plot_evaluation_metrics(y_true, preds, class_names, dataset_title_suffix):
    """
    Genera y muestra la matriz de confusión, métricas por clase (P, R, F1) y F2-score.
    """
    print(f"\n--- Resultados para: {dataset_title_suffix} ---")
    print("Accuracy:", accuracy_score(y_true, preds))
    print("F1-macro:", f1_score(y_true, preds, average="macro"))
    
    report_dict = classification_report(y_true, 
                                        preds,
                                        target_names=class_names,
                                        output_dict=True,
                                        zero_division=0)
    num_classes_plot = len(class_names)

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_true, 
                          preds,
                          labels=list(range(num_classes_plot)))
    plt.figure(figsize=(5, 4) if num_classes_plot <=2 else (6,5))
    plt.imshow(cm, 
               interpolation="nearest",
               cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusión ({dataset_title_suffix})")
    plt.colorbar()
    plt.xticks(np.arange(num_classes_plot),
               class_names,
               rotation=45, 
               ha="right")
    plt.yticks(np.arange(num_classes_plot), class_names)
    threshold = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], 
                     ha="center", 
                     va="center",
                     color="white" if cm[i, j] > threshold else "black")
    plt.ylabel("Etiqueta verdadera")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    plt.show()

    # 2. Gráfica de Precision, Recall y F1‑Score por clase
    valid_class_labels = [lbl for lbl in class_names if
                          lbl in report_dict and 
                          isinstance(report_dict[lbl], dict)]
    
    if not valid_class_labels:
        print("No hay datos de clases válidas para graficar Precision, Recall, F1-Score.")
    else:
        precision_vals = [report_dict[lbl]['precision'] for lbl in valid_class_labels]
        recall_vals    = [report_dict[lbl]['recall'] for lbl in valid_class_labels]
        f1_vals        = [report_dict[lbl]['f1-score'] for lbl in valid_class_labels]

        x = np.arange(len(valid_class_labels))
        bar_width = 0.25

        plt.figure(figsize=(7, 5) if num_classes_plot <=2 else (8,5))
        bars1 = plt.bar(x, precision_vals, bar_width, label='Precision')
        bars2 = plt.bar(x + bar_width, recall_vals, bar_width, label='Recall')
        bars3 = plt.bar(x + 2 * bar_width, f1_vals, bar_width, label='F1-Score')

        plt.xlabel('Clases')
        plt.ylabel('Puntuación')
        plt.title(f'Métricas por Clase ({dataset_title_suffix})')
        plt.xticks(x + bar_width, valid_class_labels)
        plt.legend()
        plt.ylim(0, 1.1)

        for bars_group in [bars1, bars2, bars3]:
            for bar in bars_group:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., 
                         height + 0.01, 
                         f"{height:.2f}",
                         ha='center', 
                         va='bottom',
                         fontsize=9)
        plt.tight_layout()
        plt.show()

    # 3. Gráfica de F2‑score por clase
    if not valid_class_labels:
        print("No hay datos de clases válidas para graficar F2-Score.")
    else:
        f2_vals = []
        for c in valid_class_labels:
            prec = report_dict[c]["precision"]
            rec = report_dict[c]["recall"]
            f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0 
            f2_vals.append(f2)

        plt.figure(figsize=(6, 4) if num_classes_plot <=2 else (7,4))
        bar_f2 = plt.bar(valid_class_labels, f2_vals, color="purple")
        plt.ylim(0, 1.1)
        plt.title(f"F2‑score por clase ({dataset_title_suffix})")
        plt.ylabel("F2‑score")
        for bar, val in zip(bar_f2, f2_vals):
            plt.text(bar.get_x() + bar.get_width()/2, 
                     val + 0.02, 
                     f"{val:.2f}",
                     ha="center",
                     va="bottom")
        plt.tight_layout()
        plt.show()

# --------------------------------------------------------------------------
# Escenario 1: Evaluación con Dataset Binario Balanceado (Negativo vs. Positivo)
# --------------------------------------------------------------------------
print("\n\n--- ESCENARIO 1: DATASET BINARIO BALANCEADO (NEGATIVO VS. POSITIVO) ---")

negative_label_str = 'negative'
positive_label_str = 'positive'
binary_class_names_ordered = [negative_label_str, positive_label_str] 

original_negative_id = data_original.features['label'].str2int(negative_label_str)
original_positive_id = data_original.features['label'].str2int(positive_label_str)
original_int2str_fn = data_original.features['label'].int2str

negative_samples_orig = data_original.filter(lambda ex: ex['label'] == original_negative_id)
num_negative_samples = len(negative_samples_orig)
print(f"Número de muestras '{negative_label_str}' encontradas: {num_negative_samples}")

positive_samples_all_orig = data_original.filter(lambda ex: ex['label'] == original_positive_id)
num_positive_all_samples = len(positive_samples_all_orig)
print(f"Número total de muestras '{positive_label_str}' encontradas: {num_positive_all_samples}")

data_balanced_binary = None
if num_positive_all_samples > 0 and num_negative_samples > 0:
    target_samples_for_balanced = min(num_positive_all_samples,
                                      num_negative_samples)
    print(f"Creando dataset binario balanceado con {target_samples_for_balanced} muestras por clase.")

    selected_negative_samples = negative_samples_orig.shuffle(seed=42).select(range(target_samples_for_balanced))
    selected_positive_samples = positive_samples_all_orig.shuffle(seed=42).select(range(target_samples_for_balanced))
    
    data_balanced_binary_raw = concatenate_datasets([selected_negative_samples,
                                                     selected_positive_samples])
    balanced_texts_list = [ex['text'] for ex in data_balanced_binary_raw]
    balanced_labels_str_list = [original_int2str_fn(ex['label']) for ex in data_balanced_binary_raw]

    data_balanced_binary = create_binary_hf_dataset(balanced_texts_list,
                                                    balanced_labels_str_list,
                                                    binary_class_names_ordered)
    print(f"Dataset binario balanceado creado con {len(data_balanced_binary)} muestras.")
    print("Distribución en el dataset binario balanceado:")
    for class_id_bin_bal in range(len(data_balanced_binary.features["label"].names)):
        class_name_str_bin_bal = data_balanced_binary.features["label"].int2str(class_id_bin_bal)
        count_bin_bal = len(data_balanced_binary.filter(lambda x: x['label'] == class_id_bin_bal))
        print(f"- {class_name_str_bin_bal}: {count_bin_bal}")
else:
    print("No hay suficientes muestras de 'positive' o 'negative' para crear el dataset balanceado.")


if data_balanced_binary and len(data_balanced_binary) > 0 :
    # Usar la nueva función de inferencia con elección forzada
    y_true_bal, preds_bal = perform_binary_inference_forced_choice(data_balanced_binary,
                                                                   sentiment_pipeline,
                                                                   binary_class_names_ordered)
    plot_evaluation_metrics(y_true_bal,
                            preds_bal,
                            data_balanced_binary.features["label"].names,
                            "Binario Balanceado (Forzado P/N)")
else:
    print("Evaluación del dataset binario balanceado omitida por falta de datos.")


# --------------------------------------------------------------------------
# Escenario 2: Evaluación con Dataset Binario Completo (Todos los Negativos vs. Todos los Positivos)
# --------------------------------------------------------------------------
print("\n\n--- ESCENARIO 2: DATASET BINARIO COMPLETO (TODOS NEGATIVOS VS. TODOS POSITIVOS) ---")

data_full_binary = None
if num_positive_all_samples > 0 and num_negative_samples > 0:
    print(f"Creando dataset binario completo con {len(negative_samples_orig)} '{negative_label_str}' y {len(positive_samples_all_orig)} '{positive_label_str}' muestras.")
    data_full_binary_raw = concatenate_datasets([negative_samples_orig,
                                                 positive_samples_all_orig])
    full_texts_list = [ex['text'] for ex in data_full_binary_raw]
    full_labels_str_list = [original_int2str_fn(ex['label']) for ex in data_full_binary_raw]

    data_full_binary = create_binary_hf_dataset(full_texts_list,
                                                full_labels_str_list, 
                                                binary_class_names_ordered)
    print(f"Dataset binario completo creado con {len(data_full_binary)} muestras.")
    print("Distribución en el dataset binario completo:")
    for class_id_bin_full in range(len(data_full_binary.features["label"].names)):
        class_name_str_bin_full = data_full_binary.features["label"].int2str(class_id_bin_full)
        count_bin_full = len(data_full_binary.filter(lambda x: x['label'] == class_id_bin_full))
        print(f"- {class_name_str_bin_full}: {count_bin_full}")
else:
    print("No hay suficientes muestras de 'positive' o 'negative' para crear el dataset binario completo.")


if data_full_binary and len(data_full_binary) > 0:
    # Usar la nueva función de inferencia con elección forzada
    y_true_full, preds_full = perform_binary_inference_forced_choice(data_full_binary, sentiment_pipeline, binary_class_names_ordered)
    plot_evaluation_metrics(y_true_full, preds_full, data_full_binary.features["label"].names, "Binario Completo (Forzado P/N)")
else:
    print("Evaluación del dataset binario completo omitida por falta de datos.")


print("\n--- Fin de los Escenarios de Evaluación ---")


# --------------------------------------------------------------------------
# Sección de Predicción Interactiva (también con elección forzada P/N)
# --------------------------------------------------------------------------
print("\n\n--- PREDICCIÓN INTERACTIVA DE SENTIMIENTO (FORZADO POSITIVO/NEGATIVO) ---")
print("Escribe una noticia (o 'exit' para salir):")

while True:
    user_input = input("> ")
    if user_input.lower() == 'exit':
        print("Saliendo del modo interactivo.")
        break
    if not user_input.strip():
        print("Por favor, ingresa algún texto.")
        continue

    prediction_scores_list = sentiment_pipeline(user_input)[0]

    positive_score_original = 0.0
    negative_score_original = 0.0
    neutral_score_original = 0.0 

    for item in prediction_scores_list:
        label = item['label'].lower()
        score = item['score']
        if label == positive_label_str: # Usar las variables definidas
            positive_score_original = score
        elif label == negative_label_str: # Usar las variables definidas
            negative_score_original = score
        elif label == 'neutral': 
            neutral_score_original = score

    sum_positive_negative = positive_score_original + negative_score_original
    final_sentiment_str = ""
    relative_confidence = 0.0

    if sum_positive_negative == 0:
        # Por defecto, si ambos P/N son 0, elegimos la primera clase de binary_class_names_ordered (negative)
        final_sentiment_str = binary_class_names_ordered[0].capitalize() + " (scores P/N originales eran 0, modelo muy neutral)"
        relative_confidence = 0.0 
    else:
        normalized_positive = positive_score_original / sum_positive_negative
        # normalized_negative = negative_score_original / sum_positive_negative # No es estrictamente necesaria para la decisión

        if normalized_positive >= 0.5: # Si positivo es >= 50% de la probabilidad relativa P/N
            final_sentiment_str = positive_label_str.capitalize()
            relative_confidence = normalized_positive
        else:
            final_sentiment_str = negative_label_str.capitalize()
            relative_confidence = 1.0 - normalized_positive # O negative_score_original / sum_positive_negative
    
    print(f"Sentimiento (Forzado P/N): {final_sentiment_str} (Confianza relativa P/N: {relative_confidence:.2f})")
    print("  Scores originales del modelo:")
    print(f"    {positive_label_str.capitalize()}: {positive_score_original:.4f}")
    print(f"    {negative_label_str.capitalize()}: {negative_score_original:.4f}")
    print(f"    Neutral:  {neutral_score_original:.4f}")
    print("-" * 30)

print("\n--- Fin del Script ---")