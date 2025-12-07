#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena un LightGBM binario para fraude, guarda métricas + gráficas + modelo.
"""

import pandas as pd, numpy as np, joblib, os, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, classification_report)

from lightgbm import LGBMClassifier

# ==== Funciones auxiliares para NO depender de scikit-plot ====================
def plot_cumulative_gain(y_true, y_proba, title, filename):
    """
    Dibuja la curva de Gain (ganancia acumulada) con matplotlib.
    y_true  : array-like de etiquetas reales (0/1)
    y_proba : probabilidades estimadas de la clase positiva
    title   : título de la figura
    filename: ruta donde guardar la imagen
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    # Ordenar por probabilidad descendente
    order = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[order]

    # % de fraudes captados y % de muestras investigadas
    gains = np.cumsum(y_true_sorted) / y_true.sum()
    perc_samples = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)

    plt.figure()
    plt.plot(perc_samples * 100, gains * 100, label='Gain curve')
    plt.plot([0, 100], [0, 100], '--', label='Random')
    plt.xlabel('% de transacciones investigadas')
    plt.ylabel('% fraudes detectados')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()


def plot_confusion(cm, title, filename, cmap='Greens'):
    """
    Grafica una matriz de confusión simple (sin scikit-plot).
    cm       : matriz de confusión 2x2 (numpy array)
    title    : título de la figura
    filename : ruta donde guardar
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Pred 0', 'Pred 1'])
    plt.yticks(tick_marks, ['Real 0', 'Real 1'])
    # Escribir los valores dentro de cada celda
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close()
# ============================================================================== 

# Paso 1 · Carga y pre-procesado --------------------------------------------
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1);  y = df['Class']
X['Amount'] = StandardScaler().fit_transform(X[['Amount']])

# Paso 2 · Split (reusa si existe) ------------------------------------------
split_path = 'modelos/test_split.pkl'
if os.path.exists(split_path):
    X_train, X_test, y_train, y_test = joblib.load(split_path)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42)
    os.makedirs('modelos', exist_ok=True)
    joblib.dump((X_train, X_test, y_train, y_test), split_path)

# Paso 3 · Definir modelo LightGBM ------------------------------------------
scale_pos = (len(y_train)-y_train.sum())/y_train.sum()
lgbm = LGBMClassifier(
        n_estimators=600, num_leaves=64, max_depth=-1,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        objective='binary', class_weight='balanced',
        scale_pos_weight=scale_pos, random_state=42)

print("Entrenando LightGBM …")
lgbm.fit(X_train, y_train)

# Paso 4 · Métricas y gráficas (análogo a XGBoost) ---------------------------
probas = lgbm.predict_proba(X_test)[:,1]; preds = (probas>=0.50).astype(int)
roc_auc = roc_auc_score(y_test, probas); pr_auc = average_precision_score(y_test, probas)
cm = confusion_matrix(y_test, preds)
print(classification_report(y_test, preds, digits=4))
print(f"AUC-ROC={roc_auc:.6f}  AP={pr_auc:.6f}  CM={cm.ravel()}")

os.makedirs('imagenes', exist_ok=True)
# ROC
fpr,tpr,_ = roc_curve(y_test, probas); plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.title("ROC – LightGBM"); plt.legend([f"AUC={roc_auc:.3f}"]); plt.savefig('imagenes/roc_lgbm.png'); plt.close()
# PR
prec,rec,_=precision_recall_curve(y_test,probas); plt.figure(); plt.plot(rec,prec)
plt.title("PR – LightGBM"); plt.legend([f"AP={pr_auc:.3f}"]); plt.savefig('imagenes/pr_lgbm.png'); plt.close()
# Gain & matriz de confusión (sin scikit-plot)
plot_cumulative_gain(y_test.values, probas, "Gain – LightGBM", 'imagenes/gain_lgbm.png')
plot_confusion(cm, "Confusión – LightGBM", 'imagenes/confmat_lgbm.png', cmap='Greens')
# Learning curve
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
train_sizes, tr, val = learning_curve(lgbm,X_train,y_train,cv=cv,scoring='roc_auc',
                                      train_sizes=np.linspace(0.1,1.0,6),n_jobs=-1)
plt.figure(); plt.plot(train_sizes,tr.mean(1),'o-',label="Train AUC")
plt.plot(train_sizes,val.mean(1),'o-',label="CV AUC"); plt.legend()
plt.title("Learning Curve – LightGBM"); plt.xlabel("Muestras entrenamiento")
plt.savefig('imagenes/learning_curve_lgbm.png'); plt.close()
# Feature importance
imp = pd.Series(lgbm.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(); imp.plot(kind='barh'); plt.title("Top-10 features – LightGBM"); plt.xlabel("Gain")
plt.tight_layout(); plt.savefig('imagenes/feature_importance_lgbm.png'); plt.close()

# Paso 5 · Guardar modelo ----------------------------------------------------
joblib.dump(lgbm,'modelos/lightgbm.pkl');  print("✅ LightGBM listo.\n")