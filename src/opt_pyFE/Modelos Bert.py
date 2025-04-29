# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 14:43:44 2025

@author: betoh
"""
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ----------------------------
# 1) Cargar datos y procesarlos
# ----------------------------
csv_path = r"C:\Users\betoh\OneDrive\Escritorio\Yo\Economía\Phyton\webscrapping\Analisis de sentimiento\2 datasets de finanzas.csv"

data = load_dataset(
    "csv", data_files=csv_path,
    encoding="latin-1", column_names=["label", "text"]
)["train"].class_encode_column("label")

# Usamos todo el dataset para evaluar
test_txt = data["text"]
y_true   = data["label"]
classes  = data.features["label"].names  # Lista de nombres de clases
num_classes = len(classes)
label2id = {n: i for i, n in enumerate(classes)}

# ----------------------------
# 2) Configurar pipelines (modelos)
# ----------------------------
models = [
    #"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
     "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    # "soleimanian/financial-roberta-large-sentiment"
]

pipes = [
    pipeline("text-classification", model=m, device=-1,
             return_all_scores=True, truncation=True, padding=True, batch_size=32)
    for m in models
]

# ----------------------------
# 3) Inferencia: ensamblaje de predicciones
# ----------------------------
# Sumamos las probabilidades para cada clase
logits_sum = np.zeros((len(test_txt), num_classes))
for pl in pipes:
    results = pl(test_txt)
    for i, res in enumerate(results):
        for d in res:
            logits_sum[i, label2id[d["label"]]] += d["score"]

preds = logits_sum.argmax(axis=1)

# Métricas globales
print("Ensemble Accuracy :", accuracy_score(y_true, preds))
print("Ensemble F1‑macro :", f1_score(y_true, preds, average="macro"))

# Generar reporte de clasificación (se pasan los nombres de las clases)
report_dict = classification_report(y_true, preds, target_names=classes, output_dict=True)

# ----------------------------
# 4) Gráficas de Evaluación
# ----------------------------

# 4.1. Matriz de Confusión
# Usamos índices (0, 1, ..., num_classes-1) ya que y_true y preds son numéricos
cm = confusion_matrix(y_true, preds, labels=list(range(num_classes)))
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.colorbar()
plt.xticks(np.arange(num_classes), classes)
plt.yticks(np.arange(num_classes), classes)
threshold = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i, j] > threshold else "black")
plt.ylabel("Etiqueta verdadera")
plt.xlabel("Etiqueta predicha")
plt.tight_layout()
plt.show()

# 4.2. Gráfica de Precision, Recall y F1‑Score por clase
class_labels = [lbl for lbl in classes if lbl in report_dict]
precision_vals = [report_dict[lbl]['precision'] for lbl in class_labels]
recall_vals    = [report_dict[lbl]['recall'] for lbl in class_labels]
f1_vals        = [report_dict[lbl]['f1-score'] for lbl in class_labels]

x = np.arange(len(class_labels))
bar_width = 0.25

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x, precision_vals, bar_width, label='Precision')
bars2 = plt.bar(x + bar_width, recall_vals, bar_width, label='Recall')
bars3 = plt.bar(x + 2 * bar_width, f1_vals, bar_width, label='F1-Score')

plt.xlabel('Clases')
plt.ylabel('Puntuación')
plt.title('Métricas por Clase')
plt.xticks(x + bar_width, class_labels)
plt.legend()

# Agregar data labels para cada métrica
for i, bar in enumerate(bars1):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{precision_vals[i]:.2f}",
             ha='center', va='bottom')
for i, bar in enumerate(bars2):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{recall_vals[i]:.2f}",
             ha='center', va='bottom')
for i, bar in enumerate(bars3):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{f1_vals[i]:.2f}",
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 4.3. Gráfica de F2‑score por clase
# Fórmula: F2 = (5 * precision * recall) / (4 * precision + recall) (si el denominador no es cero)
f2_vals = []
for c in classes:
    prec = report_dict[c]["precision"]
    rec = report_dict[c]["recall"]
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) else 0
    f2_vals.append(f2)

plt.figure(figsize=(7, 4))
bar_f2 = plt.bar(classes, f2_vals, color="purple")
plt.ylim(0, 1)
plt.title("F2‑score por clase")
plt.ylabel("F2‑score")
for bar, val in zip(bar_f2, f2_vals):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

