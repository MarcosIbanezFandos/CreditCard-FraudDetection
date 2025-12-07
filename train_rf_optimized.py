# ------------------------------------------------------------------------------
# Autor: Marcos Ibáñez Fandos
# Script: train_rf_optimized.py
# Descripción:
#   Entrena un Random Forest "optimizado" orientado a un escenario bancario de
#   detección de fraudes. Objetivo operativo: lograr más recall
#   (detectar más % de los fraudes) minimizando las falsas alarmas dentro de
#   esa restricción.
#
#   Flujo comentado PASO A PASO:
#       0. Preparación de entorno y carpetas
#       1. Carga y preprocesado del dataset
#       2. División estratificada Train/Test (70 % / 30 %)
#       3. Construcción del pipeline: SafeSMOTE  + RandomForest
#       4. Búsqueda de hiper-parámetros con HalvingGridSearchCV (scoring=recall)
#       5. Ajuste final del mejor modelo con todo el train
#       6. Evaluación final con el umbral por defecto (0.50)
#       7. Evaluación completa (métricas y gráficas)
#       8. Serialización del modelo y split de test
# ------------------------------------------------------------------------------

# Paso 0. Preparación -----------------------------------------------------------------------
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activamos explícitamente la API experimental de HalvingGridSearchCV
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

#   Creamos carpetas de salida para modelos y gráficas
os.makedirs('modelos', exist_ok=True)
os.makedirs('imagenes', exist_ok=True)

print("===== Random Forest OPTIMIZADO para fraude =====")

# Paso 1. Carga y preprocesado --------------------------------------------------------------
print("Paso 1: Cargando creditcard.csv y estandarizando 'Amount'…")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1).copy()
y = df['Class'].copy()

scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# Paso 2. División Train/Test ---------------------------------------------------------------
print("Paso 2: Dividiendo en train/test (70-30, estratificado)…")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Paso 3. Pipeline SMOTE + RandomForest ------------------------------------------------------
print("Paso 3: Construyendo pipeline SafeSMOTE + RandomForest…")
#   SafeSMOTE: ampliamos la clase minoritaria sólo en train para equilibrar
# Usamos k_neighbors=1 para que SMOTE funcione incluso en los subconjuntos
# pequeños que genera HalvingGridSearch (donde puede haber muy pocos fraudes).
sampler = SMOTE(random_state=42, k_neighbors=1)
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
pipe = Pipeline(steps=[('smote', sampler), ('clf', rf)])

# Paso 4. Búsqueda de hiper-parámetros -------------------------------------------------------
print("Paso 4: HalvingGridSearchCV (scoring=recall)…")
param_grid = {
    'clf__n_estimators':  [200, 400, 600],
    'clf__max_depth':     [None, 10, 20],
    'clf__min_samples_leaf': [1, 2, 5],
    'clf__class_weight':  [None, 'balanced']
}
grid = HalvingGridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    factor=2,
    cv=3,
    scoring='recall',   # maximizamos recall en CV
    n_jobs=-1,
    verbose=1
)

#   Para acelerar, usamos sólo el 30 % del train dentro de la búsqueda
X_tune, _, y_tune, _ = train_test_split(
    X_train, y_train, test_size=0.7, stratify=y_train, random_state=42
)

print("    > Ajustando HalvingGridSearch (sub-muestra del train)…")
grid.fit(X_tune, y_tune)
print("    > Mejores hiper-parámetros:")
print(grid.best_params_)

# Paso 5. Re-entrenar pipeline ganador en TODO el train -------------------------------------
print("Paso 5: Re-entrenando mejor modelo sobre el 70 % completo…")
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

# Paso 6. Evaluación final con el umbral por defecto (0.50) -------------------
print("Paso 6: Evaluación final usando el umbral por defecto (0.50)…")
y_pred = best_model.predict(X_test)

# Paso 7. Evaluación final -------------------------------------------------------------------
print("Paso 7: Métricas finales usando el umbral por defecto…")

print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
ap  = average_precision_score(y_test, best_model.predict_proba(X_test)[:, 1])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"AUC-ROC: {auc:.6f}")
print(f"Average Precision (PR AUC): {ap:.4f}")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# >>> Gráficas -----------------------------------------------------------------------------
print("    > Guardando gráficas en 'imagenes/'…")
# 7.1 ROC
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(); plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC – RF Optimizado'); plt.legend(); plt.savefig('imagenes/roc_rf_optim.png'); plt.close()

# 7.2 PR
precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(); plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR – RF Optimizado')
plt.legend(); plt.savefig('imagenes/pr_rf_optim.png'); plt.close()

# 7.3 Matriz de confusión (heatmap)
plt.figure(); cm = np.array([[tn, fp],[fn, tp]])
plt.imshow(cm, cmap='Oranges'); plt.title('Confusión – RF Optimizado')
plt.colorbar(); plt.xticks([0,1], ['Pred 0','Pred 1']); plt.yticks([0,1], ['Real 0','Real 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.savefig('imagenes/confmat_rf_optim.png'); plt.close()

# 7.4 Gain / Lift
proba_test = best_model.predict_proba(X_test)[:, 1]
sorted_idx = np.argsort(-proba_test)
y_sorted = y_test.values[sorted_idx]
cum_fraud = np.cumsum(y_sorted); total_fraud = cum_fraud[-1]
pct_tx = np.arange(1, len(y_sorted)+1) / len(y_sorted)
lift = cum_fraud / total_fraud
plt.figure(); plt.plot(pct_tx*100, lift*100, label='Gain curve')
plt.plot([0,100],[0,100],'--', label='Random')
plt.xlabel('% transacciones investigadas'); plt.ylabel('% fraudes detectados')
plt.title('Gain – RF Optimizado'); plt.legend(); plt.savefig('imagenes/gain_rf_optim.png'); plt.close()

# 7.5 Learning curve
train_sizes, train_scores, valid_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1,
    train_sizes=np.linspace(0.1,1.0,5), shuffle=True, random_state=42)
plt.figure()
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train AUC')
plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', label='CV AUC')
plt.xlabel('Muestras entrenamiento'); plt.ylabel('AUC-ROC')
plt.title('Learning Curve – RF Optimizado')
plt.legend(); plt.savefig('imagenes/learning_curve_rf_optim.png'); plt.close()

# 7.6 Importancia de variables
importances = best_model.named_steps['clf'].feature_importances_
feat_names = X_train.columns
idx_sorted = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(8,5))
plt.barh(range(10), importances[idx_sorted][::-1])
plt.yticks(range(10), feat_names[idx_sorted][::-1])
plt.xlabel('Importancia (MDI)'); plt.title('Top-10 features – RF Optimizado')
plt.savefig('imagenes/feature_importance_rf_optim.png'); plt.close()

# Paso 8. Serialización ----------------------------------------------------------------------
print("Paso 8: Guardando modelo y artefactos en carpeta 'modelos'…")
joblib.dump(best_model, 'modelos/rf_optimized.pkl')
#   Guardamos también el split de test
joblib.dump((X_test, y_test), 'modelos/test_split_rf.pkl')
print("-> rf_optimized.pkl y test_split_rf.pkl guardados.")

print("==== Entrenamiento y evaluación del RF optimizado completados ====")