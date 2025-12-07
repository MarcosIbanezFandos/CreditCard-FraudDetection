import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

def ejecutar_simulacion(
    modo,
    C_FN,
    C_FP_auto,
    C_FP_manual,
    T_low,
    markup_micro,
    markup_med,
    markup_alto,
    min_rec_high,
    min_rec_all,
    min_prec_all,
    high_amt_never_pass,
    nombre
):
    # 1. Cargar modelo (ajusta la ruta si es necesario)
    model = joblib.load("modelos/rf_optimized.pkl")

    # 2. Cargar dataset de test
    df = pd.read_csv("datos/creditcard.csv")  # Asegúrate de que esta ruta es correcta

    # 3. Preparar variables
    X = df.drop(columns=["Class", "Amount"])
    y_true = df["Class"]

    # 4. Predecir scores y clases con threshold provisional
    y_scores = model.predict_proba(X)[:, 1]
    threshold = 0.5  # ← puedes ajustar esto si quieres evaluar diferentes umbrales
    y_pred = (y_scores >= threshold).astype(int)

    # 5. Calcular métricas
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 6. Mostrar resultados
    print(f"--- Resultados para {nombre} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 7. Retornar métricas si se usa desde otro módulo (como Streamlit)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }