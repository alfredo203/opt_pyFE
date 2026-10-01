"""Módulo para análisis de sentimiento en noticias financieras mediante FinancialBERT."""

from typing import List, Union
import pandas as pd


def analizar_sentimiento(
    textos: Union[str, List[str]],
    model_name: str = "ahmedrachid/FinancialBERT-Sentiment-Analysis",
    forzar_binario: bool = True,
    device: int = -1,
) -> pd.DataFrame:
    """Clasifica el sentimiento de titulares financieros usando un modelo FinancialBERT.

    Parameters
    ----------
    textos : str or list of str
        Texto o lista de textos (titulares/noticias) a analizar.
    model_name : str, default="ahmedrachid/FinancialBERT-Sentiment-Analysis"
        Identificador del modelo en Hugging Face.
    forzar_binario : bool, default=True
        Si es True, fuerza la clasificación a Positivo o Negativo evaluando el score mayor entre ambos.
    device : int, default=-1
        Dispositivo para inferencia (-1 para CPU, >=0 para GPU CUDA).

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: ['Noticia', 'Clasificacion', 'Score_Positivo', 'Score_Negativo', 'Score_Neutral'].
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "La librería 'transformers' es requerida para el análisis de sentimiento.\n"
            "Instálala con: pip install opt_pyFE[nlp] o pip install transformers torch"
        )

    if isinstance(textos, str):
        textos_lista = [textos]
    else:
        textos_lista = list(textos)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=model_name,
        device=device,
        return_all_scores=True,
        truncation=True,
        padding=True,
    )

    filas = []
    for txt in textos_lista:
        if not txt.strip():
            continue

        raw_scores = sentiment_pipeline(txt)[0]
        score_pos = 0.0
        score_neg = 0.0
        score_neu = 0.0

        for item in raw_scores:
            lbl = item["label"].lower()
            val = float(item["score"])
            if "pos" in lbl:
                score_pos = val
            elif "neg" in lbl:
                score_neg = val
            elif "neu" in lbl:
                score_neu = val

        if forzar_binario:
            clasificacion = "Positive" if score_pos >= score_neg else "Negative"
        else:
            max_score = max(score_pos, score_neg, score_neu)
            if max_score == score_pos:
                clasificacion = "Positive"
            elif max_score == score_neg:
                clasificacion = "Negative"
            else:
                clasificacion = "Neutral"

        filas.append(
            {
                "Noticia": txt,
                "Clasificacion": clasificacion,
                "Score_Positivo": round(score_pos, 4),
                "Score_Negativo": round(score_neg, 4),
                "Score_Neutral": round(score_neu, 4),
            }
        )

    return pd.DataFrame(filas)


# Alias amigable
analyze_sentiment = analizar_sentimiento
