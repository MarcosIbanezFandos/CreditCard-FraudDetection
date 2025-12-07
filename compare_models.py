#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparación de 5 modelos de fraude ya entrenados (RF baseline, RF optimizado, XGBoost, LightGBM, LogisticReg).

Autor: Marcos Ibáñez Fandos 
Fecha: 2025-07-22

Descripción:
    - Carga los modelos y el split de test guardados previamente (.pkl).
    - Permite fijar un umbral de decisión común para todos (--threshold 0.4) o,
      si no se indica, calcula para cada modelo el umbral que maximiza el F1-score.
    - Calcula métricas (precision, recall, F1, ROC-AUC, PR-AUC, etc.) y las guarda en CSV.
    - Genera 5 gráficas comparativas y las almacena en /comparar:
        1) ROC (todas las curvas)
        2) Precision-Recall (todas las curvas)
        3) Barras de métricas (Precision, Recall, F1)
        4) Gain/Lift comparativa
        5) Matrices de confusión en grid o separadas (una por modelo)

Uso:
    python compare_models.py               # modo AUTO: umbral óptimo por F1 para cada modelo
    python compare_models.py --threshold 0.4

Requisitos:
    - Tener los ficheros .pkl en la carpeta 'modelos/' (ajusta paths si hace falta):
        rf_baseline.pkl, rf_optimized.pkl, xgboost.pkl, lightgbm.pkl, logreg.pkl
      y el split de test 'test_split.pkl' (X_test, y_test).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, f1_score, precision_score, recall_score)

# ---- Compatibility shim for scikit-learn 1.7.x and imblearn ----
try:
    import sklearn.utils._tags as _sk_tags
    if not hasattr(_sk_tags, "_safe_tags"):
        # Provide stub for imblearn compatibility
        def _safe_tags(estimator, key=None):
            return {}
        _sk_tags._safe_tags = _safe_tags
except ImportError:
    pass
# -----------------------------------------------------------------

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# --------------------  Funciones auxiliares  --------------------

def ensure_dir(path):
    """Crea el directorio si no existe."""
    os.makedirs(path, exist_ok=True)

def best_threshold_f1(y_true, y_proba):
    """Devuelve el umbral que maximiza F1, junto con F1, precision y recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Nota: thresholds tiene len = len(precisions)-1
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
    idx = np.argmax(f1_scores)
    return thresholds[idx], f1_scores[idx], precisions[idx], recalls[idx]

def cumulative_gain_curve(y_true, y_proba):
    """
    Calcula la curva de Gain manualmente.
    Devuelve:
        pct_samples (0..100), pct_fraudes_detectados (0..100)
    """
    # Ordenamos por probabilidad descendente
    order = np.argsort(-y_proba)
    y_sorted = np.array(y_true)[order]

    total_fraudes = y_sorted.sum()
    cum_fraudes = np.cumsum(y_sorted)
    pct_fraudes = cum_fraudes / (total_fraudes + 1e-12)

    n = len(y_sorted)
    pct_muestras = np.arange(1, n+1) / n
    return pct_muestras * 100, pct_fraudes * 100

def evaluate_model(name, model, X_test, y_test, threshold=None):
    """
    Evalúa un modelo dado.
    - Si threshold es None, busca el umbral que maximiza F1.
    - Devuelve un diccionario con métricas + curvas para gráficos.
    """
    # 1. Probabilidades para la clase 1
    proba = model.predict_proba(X_test)[:, 1]

    # 2. Selección de umbral
    if threshold is None:
        thr, f1_opt, prec_opt, rec_opt = best_threshold_f1(y_test, proba)
        chosen_thr = thr
    else:
        chosen_thr = threshold
        # calculamos F1 con ese umbral únicamente
        preds_tmp = (proba >= chosen_thr).astype(int)
        f1_opt = f1_score(y_test, preds_tmp)
        prec_opt = precision_score(y_test, preds_tmp, zero_division=0)
        rec_opt  = recall_score(y_test, preds_tmp,  zero_division=0)

    # 3. Predicciones con el umbral elegido
    preds = (proba >= chosen_thr).astype(int)

    # 4. Métricas
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    # 5. Curvas para ROC y PR
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, proba)

    # 6. Curva Gain
    pct_samples, pct_gain = cumulative_gain_curve(y_test, proba)

    return {
        "name": name,
        "threshold": chosen_thr,
        "precision": prec_opt,
        "recall": rec_opt,
        "f1": f1_opt,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr_curve": fpr, "tpr_curve": tpr,
        "prec_curve": prec_curve, "rec_curve": rec_curve,
        "pct_samples": pct_samples, "pct_gain": pct_gain
    }

# --------------------  MAIN  --------------------

def main():
    parser = argparse.ArgumentParser(description="Comparar modelos de fraude.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral común para todos los modelos. Si se omite, se optimiza F1 por modelo.")
    parser.add_argument("--outdir", type=str, default="comparar",
                        help="Carpeta donde guardar resultados y gráficas.")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # --- Sufijo/nombre según el umbral elegido ---
    if args.threshold is None:
        suffix = "_autoF1"  # cada modelo usa el umbral que maximiza su F1
        title_extra = " (umbrales óptimos por F1)"
    else:
        suffix = f"_thr{args.threshold:.2f}".replace(".", "p")  # evita el punto en el nombre del fichero
        title_extra = f" (umbral común = {args.threshold:.2f})"

    # 1. Paths de modelos y split
    model_dir = "modelos"
    split_path_candidates = [
        os.path.join(model_dir, "test_split.pkl"),        # tu último nombre
        os.path.join(model_dir, "test_split_rf.pkl"),
        os.path.join(model_dir, "test_split_baseline.pkl")
    ]
    split_path = None
    for p in split_path_candidates:
        if os.path.exists(p):
            split_path = p
            break
    if split_path is None:
        raise FileNotFoundError("No encuentro el archivo test_split.pkl (o similares) en la carpeta 'modelos/'.")
    data = joblib.load(split_path)
    # El archivo de split puede haberse guardado como tupla (2 o 4 elementos) o como diccionario.
    if isinstance(data, tuple):
        if len(data) == 4:
            # Formato típico: (X_train, X_test, y_train, y_test)
            _, X_test, _, y_test = data
        elif len(data) == 2:
            # Formato: (X_test, y_test)
            X_test, y_test = data
        else:
            raise ValueError(f"Formato de tupla inesperado en {split_path}: len={len(data)}")
    elif isinstance(data, dict):
        # Intentamos varias claves comunes
        X_test = data.get('X_test') or data.get('Xtest') or data.get('X_val')
        y_test = data.get('y_test') or data.get('ytest') or data.get('y_val')
        if X_test is None or y_test is None:
            raise ValueError(f"No se encontraron claves 'X_test'/'y_test' en el diccionario del split: {list(data.keys())}")
    else:
        raise TypeError(f"Tipo de objeto inesperado al cargar {split_path}: {type(data)}")

    # 2. Cargar modelos
    model_files = {
        "RF Baseline":   os.path.join(model_dir, "rf_baseline.pkl"),
        "RF Optimizado": os.path.join(model_dir, "rf_optimized.pkl"),
        "XGBoost":       os.path.join(model_dir, "xgboost.pkl"),
        "LightGBM":      os.path.join(model_dir, "lightgbm.pkl"),
        "LogisticReg":   os.path.join(model_dir, "logreg.pkl")
    }
    modelos = {}
    for nombre, path in model_files.items():
        if not os.path.exists(path):
            print(f"⚠️  No se encontró {path}. Se omite '{nombre}'.")
            continue
        modelos[nombre] = joblib.load(path)

    if not modelos:
        raise RuntimeError("No se cargó ningún modelo. Revisa los paths.")

    # 3. Evaluar todos
    resultados = []
    for nombre, modelo in modelos.items():
        print(f"Evaluando: {nombre} ...")
        res = evaluate_model(nombre, modelo, X_test, y_test, threshold=args.threshold)
        resultados.append(res)

    # 4. Guardar tabla de métricas
    df = pd.DataFrame([{
        "Modelo": r["name"],
        "Umbral": round(r["threshold"], 4),
        "Precision": round(r["precision"], 3),
        "Recall": round(r["recall"], 3),
        "F1": round(r["f1"], 3),
        "ROC_AUC": round(r["roc_auc"], 3),
        "PR_AUC": round(r["pr_auc"], 3),
        "TN": r["tn"], "FP": r["fp"], "FN": r["fn"], "TP": r["tp"]
    } for r in resultados])

    csv_path = os.path.join(args.outdir, f"metrics_comparativa{suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTabla de métricas guardada en: {csv_path}\n")
    print(df.to_string(index=False))
    print(f"→ Sufijo usado para archivos: '{suffix}'{title_extra}")

    # 5. Gráficas comparativas
    # 5.1 ROC
    plt.figure(figsize=(7,5))
    for r in resultados:
        plt.plot(r["fpr_curve"], r["tpr_curve"],
                 label=f"{r['name']} (AUC={r['roc_auc']:.3f}, thr={r['threshold']:.3f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC - Comparativa modelos" + title_extra)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"roc_comparativa{suffix}.png"), dpi=150)
    plt.close()

    # 5.2 Precision-Recall
    plt.figure(figsize=(7,5))
    for r in resultados:
        plt.plot(r["rec_curve"], r["prec_curve"],
                 label=f"{r['name']} (AP={r['pr_auc']:.3f}, thr={r['threshold']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall - Comparativa modelos" + title_extra)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"pr_comparativa{suffix}.png"), dpi=150)
    plt.close()

    # 5.3 Barras de métricas
    width = 0.25
    models_names = [r["name"] for r in resultados]
    precisions = [r["precision"] for r in resultados]
    recalls    = [r["recall"] for r in resultados]
    f1s        = [r["f1"] for r in resultados]

    x = np.arange(len(models_names))
    plt.figure(figsize=(9,5))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x,         recalls,    width, label="Recall")
    plt.bar(x + width, f1s,        width, label="F1")
    plt.xticks(x, models_names, rotation=15)
    plt.ylim(0,1.05)
    plt.ylabel("Valor")
    plt.title("Precision / Recall / F1 por modelo" + title_extra)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"metrics_barras{suffix}.png"), dpi=150)
    plt.close()

    # 5.4 Gain/Lift comparativa (Gain)
    plt.figure(figsize=(7,5))
    for r in resultados:
        plt.plot(r["pct_samples"], r["pct_gain"], label=r["name"])
    # Línea aleatoria
    plt.plot([0,100],[0,100],'k--', alpha=0.4, label="Aleatorio")
    plt.xlabel("% transacciones investigadas")
    plt.ylabel("% fraudes detectados")
    plt.title("Cumulative Gain - Comparativa modelos" + title_extra)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"gain_comparativa{suffix}.png"), dpi=150)
    plt.close()

    # 5.5 Matrices de confusión (una figura grid opcional)
    fig, axes = plt.subplots(1, len(resultados), figsize=(4*len(resultados), 3))
    if len(resultados)==1:
        axes = [axes]
    for ax, r in zip(axes, resultados):
        cm = np.array([[r["tn"], r["fp"]],
                       [r["fn"], r["tp"]]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(r["name"])
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"])
        ax.set_yticklabels(["Real 0","Real 1"])
        for (i,j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center', color='black', fontsize=9)
    fig.suptitle("Matrices de confusión" + title_extra)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(args.outdir, f"confmat_grid{suffix}.png"), dpi=150)
    plt.close()

    print(f"Gráficas guardadas en carpeta: {args.outdir}")

if __name__ == "__main__":
    main()