"""Script de demostración para el análisis de sentimiento financiero con FinancialBERT.

Requiere la instalación previa de transformers y torch:
    pip install opt_pyFE[nlp]
    o
    pip install transformers torch
"""

import sys
from opt_pyFE import analizar_sentimiento


def main():
    print("=" * 60)
    print(" DEMO: ANÁLISIS DE SENTIMIENTO FINANCIERO (FinancialBERT)")
    print("=" * 60)

    titulares_ejemplo = [
        "Apple reports record quarterly revenue beating Wall Street estimates.",
        "Tesla recalls 50,000 vehicles due to software glitches and margin concerns.",
        "Federal Reserve signals potential interest rate cuts ahead in next meeting.",
        "Company XYZ files for bankruptcy protection amid rising debt defaults.",
    ]

    print("\nAnalizando titulares de ejemplo...")
    try:
        df_resultados = analizar_sentimiento(titulares_ejemplo, forzar_binario=True)
        print("\nResultados obtenidos:")
        print(df_resultados.to_string(index=False))
    except ImportError as e:
        print(f"\n[Aviso]: {e}")
        return

    # Modo interactivo opcional
    print("\n¿Deseas probar en modo interactivo? (s/n): ", end="")
    respuesta = input().strip().lower()
    if respuesta == "s":
        print("\nEscribe un titular financiero en inglés (o 'exit' para salir):")
        while True:
            txt = input("> ").strip()
            if txt.lower() in ("exit", "quit"):
                break
            if not txt:
                continue
            res = analizar_sentimiento(txt, forzar_binario=True)
            print(f"  Clasificación : {res['Clasificacion'].iloc[0]}")
            print(f"  Score Positivo: {res['Score_Positivo'].iloc[0]}")
            print(f"  Score Negativo: {res['Score_Negativo'].iloc[0]}")
            print(f"  Score Neutral : {res['Score_Neutral'].iloc[0]}\n")


if __name__ == "__main__":
    main()
