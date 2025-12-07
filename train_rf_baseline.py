# ------------------------------------------------------------------------------
# Autor: Marcos Ibáñez Fandos
# Script: train_rf_baseline.py
# Descripción:
#   Entrena un Random Forest "baseline" sobre el dataset de fraudes en tarjetas
#   (creditcard.csv). El flujo se documenta PASO A PASO:
#       1. Carga y preprocesado de datos
#       2. División estratificada Train/Test
#       3. Entrenamiento del modelo
#       4. Evaluación con múltiples métricas
#       5. Generación y guardado de gráficas clave
#       6. Serialización del modelo y del test‑split
# ------------------------------------------------------------------------------

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Paso 0. Preparamos carpetas de salida -------------------------------------------------------
#    Creamos 'modelos/' (para .pkl) y 'imagenes/' (para .png) si no existen.
os.makedirs('modelos', exist_ok=True)
os.makedirs('imagenes', exist_ok=True)

# Paso 1. Carga y preprocesado ---------------------------------------------------------------
#    • Leemos creditcard.csv completo.
#    • Separamos features (X) y variable objetivo (y).
#    • Estandarizamos la columna 'Amount' para que tenga media 0 y varianza 1
#      (mejora la estabilidad de muchos clasificadores).
print("Step 1: Cargando datos y preprocesando...")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']
# Normalizamos 'Amount' para mejorar convergencia del modelo
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Paso 2. División estratificada Train/Test --------------------------------------------------
#    • 70 % para entrenamiento, 30 % para test.
#    • Estratificado para mantener la proporción de fraudes raros.
#    • random_state=42 -> reproducibilidad.
print("Step 2: Dividiendo dataset en train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Paso 3. Entrenamiento del Random Forest ----------------------------------------------------
#    • n_estimators = 100 árboles (compromiso precisión/tiempo).
#    • random_state=42 para reproducibilidad.
#    • n_jobs=-1 usaría todos los cores, pero aquí dejamos el default.
print("Step 3: Entrenando RandomForest Baseline...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Paso 4. Evaluación en el conjunto de test --------------------------------------------------
#    Calculamos y mostramos las métricas más relevantes para un problema
#    de detección de fraudes altamente desbalanceado:
#      - Accuracy
#      - Precision (fraudes detectados/alertas totales)
#      - Recall (fraudes detectados/fraudes reales)
#      - F1‑score (media harmónica P/R)
#      - AUC‑ROC (área bajo curva ROC)
#      - Average Precision (área bajo curva Precision‑Recall)
#      - Matthews Correlation Coefficient
#      - Balanced Accuracy
#      - Specificity (Tasa de verdaderos negativos)
#    También imprimimos la matriz de confusión (TN,FP,FN,TP).
print("Step 4: Evaluando en test set...")
y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC: {auc:.6f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

accuracy = accuracy_score(y_test, y_pred)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = f1_score(y_test, y_pred)
ap = average_precision_score(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)
bal_acc = balanced_accuracy_score(y_test, y_pred)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (class 1): {precision:.4f}")
print(f"Recall (class 1): {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Average Precision (PR AUC): {ap:.4f}")
print(f"Matthews Corrcoef: {mcc:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Specificity (class 0): {specificity:.4f}")

# Paso 5. Gráficas explicativas --------------------------------------------------------------
#      5.1 Curva ROC
#      5.2 Curva Precision‑Recall
#      5.3 Heatmap de Matriz de Confusión
#      5.4 Curva de Ganancia (Lift/Cumulative Gain)
#      5.5 Curva de Aprendizaje (AUC vs tamaño de training)
#      5.6 Barras de Importancia de variables (Top‑10)

# 5.1 Curva ROC: trade‑off TPR vs FPR
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC – RF Baseline')
plt.legend(); plt.savefig('imagenes/roc_rf_baseline.png'); plt.close()

# 5.2 Curva Precision‑Recall: útil en datasets desbalanceados
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall_curve, precision_curve, label=f'PR AUC = {ap:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision‑Recall – RF Baseline')
plt.legend(); plt.savefig('imagenes/pr_rf_baseline.png'); plt.close()

# 5.3 Heatmap de la Matriz de Confusión
plt.figure()
cm = np.array([[tn, fp],[fn, tp]])
plt.imshow(cm, cmap='Blues')
plt.title('Matriz de confusión – RF Baseline')
plt.colorbar()
plt.xticks([0,1], ['Pred 0','Pred 1'])
plt.yticks([0,1], ['Real 0','Real 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.savefig('imagenes/confmat_rf_baseline.png'); plt.close()

# 5.4 Curva de Ganancia acumulada (cuántos fraudes captamos revisando x% de transacciones)
sorted_idx = np.argsort(-y_proba)          # índices ordenados por score desc.
y_sorted = y_test.values[sorted_idx]
cum_fraud = np.cumsum(y_sorted)
total_fraud = cum_fraud[-1]
pct_tx = np.arange(1, len(y_sorted)+1) / len(y_sorted)
lift = cum_fraud / total_fraud
plt.figure()
plt.plot(pct_tx*100, lift*100, label='Gain curve')
plt.plot([0,100], [0,100], '--', label='Random')
plt.xlabel('% de transacciones investigadas')
plt.ylabel('% de fraudes captados')
plt.title('Cumulative Gain – RF Baseline')
plt.legend(); plt.savefig('imagenes/gain_rf_baseline.png'); plt.close()

# 5.5 Curva de Aprendizaje: ¿beneficia entrenar con más datos?
train_sizes, train_scores, valid_scores = learning_curve(
    clf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
)
train_mean = train_scores.mean(axis=1)
valid_mean = valid_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, 'o-', label='Train AUC')
plt.plot(train_sizes, valid_mean, 'o-', label='CV AUC')
plt.xlabel('Muestras de entrenamiento')
plt.ylabel('AUC-ROC')
plt.title('Learning Curve – RF Baseline')
plt.legend(); plt.savefig('imagenes/learning_curve_rf_base.png'); plt.close()

# 5.6 Top‑10 Importancias de variables según Gini/MDI
importances = clf.feature_importances_
feat_names = X_train.columns
idx_sorted = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(8,5))
plt.barh(range(10), importances[idx_sorted][::-1])
plt.yticks(range(10), feat_names[idx_sorted][::-1])
plt.xlabel('Importancia (MDI)')
plt.title('Top‑10 features – RF Baseline')
plt.savefig('imagenes/feature_importance_rf_base.png'); plt.close()
# ---------------------------------------------------------------

# Paso 6. Serialización del modelo y del split de test ---------------------------------------
print("Step 5: Guardando modelo...")
joblib.dump(clf, 'modelos/rf_baseline.pkl')
joblib.dump((X_test, y_test), 'modelos/test_split_baseline.pkl')
print("-> rf_baseline.pkl y test_split_baseline.pkl guardados en carpeta 'modelos'")
