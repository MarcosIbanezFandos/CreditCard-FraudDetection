#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(1) Entrena un XGBoost para detección de fraude
(2) Guarda métricas, gráficas y modelo serializado
Autor:  …  Fecha: …
"""

# Paso 0 · Imports
import pandas as pd, numpy as np, joblib, os, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier

# Eliminamos la dependencia de scikit-plot para que el script sea auto-contenido
_HAS_SKPLOT = False

def plot_cumulative_gain(y_true, y_scores, title, save_path):
    """
    Dibuja la curva de Gain acumulado sin scikit-plot.
    y_true   : array/Series binaria (0/1)
    y_scores : probabilidades de la clase positiva
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    # Ordenar por score descendente
    order = np.argsort(-y_scores)
    y_sorted = y_true[order]

    cum_positives = np.cumsum(y_sorted)
    total_positives = y_true.sum()
    perc_positives = cum_positives / total_positives * 100.0

    perc_samples = np.arange(1, len(y_true) + 1) / len(y_true) * 100.0

    plt.figure()
    plt.plot(perc_samples, perc_positives, label="Gain curve")
    plt.plot([0, 100], [0, 100], "--", label="Random")
    plt.xlabel("% de transacciones investigadas")
    plt.ylabel("% de fraudes detectados")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_confusion(cm, classes, title, save_path):
    """
    Dibuja matriz de confusión con matplotlib.
    cm      : array 2x2
    classes : lista con nombres de clases
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Purples")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)

    # Anotar cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# Paso 1 · Cargar y pre-procesar dataset ------------------------------------
print("Paso 1: Cargando creditcard.csv …")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1);      y = df['Class']

# Escalar columna Amount
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Paso 2 · Split train/test (reutiliza split guardado) -----------------------
split_path = 'modelos/test_split.pkl'
if os.path.exists(split_path):
    print("   › Reutilizando split guardado.")
    X_train, X_test, y_train, y_test = joblib.load(split_path)
else:
    print("Paso 2: Dividiendo en train/test (70-30, estratificado)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    os.makedirs('modelos', exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), split_path)

# Paso 3 · Definir y entrenar el modelo  -------------------------------------
print("Paso 3: Entrenando XGBoost …")
xgb = XGBClassifier(
        n_estimators=400,          # un número razonable
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(),  # balanceo
        random_state=42,
        eval_metric='auc')

xgb.fit(X_train, y_train)

# Paso 4 · Inferencia y métricas base ----------------------------------------
print("Paso 4: Evaluando en test …")
probas = xgb.predict_proba(X_test)[:,1]
preds  = (probas >= 0.50).astype(int)            # umbral por defecto

roc_auc = roc_auc_score(y_test, probas)
pr_auc  = average_precision_score(y_test, probas)
cm      = confusion_matrix(y_test, preds)
report  = classification_report(y_test, preds, digits=4)

print(report)
print(f"AUC-ROC: {roc_auc:.6f}")
print(f"Average precision (PR AUC): {pr_auc:.6f}")
print(f"Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

# Paso 5 · Graficar ----------------------------------------------------------
os.makedirs('imagenes', exist_ok=True)

# 5.1 ROC
fpr, tpr, _ = roc_curve(y_test, probas)
plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); 
plt.xlabel("FPR"); plt.ylabel("TPR"); 
plt.title("ROC – XGBoost"); plt.legend([f"AUC = {roc_auc:.3f}"])
plt.savefig('imagenes/roc_xgb.png'); plt.close()

# 5.2 Precision–Recall
prec, rec, _ = precision_recall_curve(y_test, probas)
plt.figure(); plt.plot(rec, prec); 
plt.xlabel("Recall"); plt.ylabel("Precision");
plt.title("PR – XGBoost"); plt.legend([f"AP = {pr_auc:.3f}"])
plt.savefig('imagenes/pr_xgb.png'); plt.close()

# 5.3 Gain & Lift  ---------------------------------------------
plot_cumulative_gain(y_test, probas, "Gain – XGBoost", "imagenes/gain_xgb.png")

# 5.4 Matriz de confusión
plot_confusion(cm, ["Real 0", "Real 1"], "Confusión – XGBoost", "imagenes/confmat_xgb.png")

# 5.5 Learning curve (AUC-ROC) ---------------
print("   › Dibujando Learning curve …")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
        xgb, X_train, y_train, cv=cv, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)

plt.figure()
plt.plot(train_sizes, train_scores.mean(1), 'o-', label="Train AUC")
plt.plot(train_sizes, val_scores.mean(1),  'o-', label="CV AUC")
plt.xlabel("Muestras entrenamiento"); plt.ylabel("AUC-ROC")
plt.title("Learning Curve – XGBoost"); plt.legend()
plt.savefig('imagenes/learning_curve_xgb.png'); plt.close()

# 5.6 Importancia de características (gain basada en árboles) --------------
plt.figure()
importances = xgb.get_booster().get_score(importance_type='gain')
imp_series = pd.Series(importances).sort_values(ascending=False).head(10)
imp_series[::-1].plot(kind='barh')         # orden ascendente
plt.xlabel("Importancia (Gain)"); plt.title("Top-10 features – XGBoost")
plt.tight_layout(); plt.savefig('imagenes/feature_importance_xgb.png'); plt.close()

# Paso 6 · Serializar modelo -------------------------------------------------
print("Paso 6: Guardando modelo en ‘modelos/xgboost.pkl’ …")
joblib.dump(xgb, 'modelos/xgboost.pkl')

print("✅ XGBoost completo.\n")