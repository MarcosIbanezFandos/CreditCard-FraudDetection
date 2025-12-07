#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena una Regresión Logística con penalización L2 (baseline lineal).
"""

import pandas as pd, numpy as np, joblib, os, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, roc_curve,
                             confusion_matrix, classification_report)

# === Funciones auxiliares para no depender de scikit-plot =====================
def plot_cumulative_gain(y_true, y_scores, filepath):
    """
    Dibuja la curva de Cumulative Gain sin scikit-plot.
    y_true  : array de etiquetas reales (0/1)
    y_scores: probabilidades para la clase positiva
    filepath: ruta donde guardar la figura
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Ordenamos por probabilidad descendente
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[order]

    total_positives = y_true_sorted.sum()
    cum_positives = np.cumsum(y_true_sorted)

    n = len(y_true_sorted)
    pct_samples = np.arange(1, n + 1) / n
    pct_positives = cum_positives / total_positives

    plt.figure()
    plt.plot(pct_samples, pct_positives, label='Cumulative Gain')
    plt.plot([0, 1], [0, 1], '--', label='Aleatorio')
    plt.xlabel('Porcentaje de muestras')
    plt.ylabel('Porcentaje de positivos acumulados')
    plt.title('Curva de Cumulative Gain')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def plot_confusion(cm, classes, filepath, normalize=False, cmap='Blues'):
    """
    Dibuja una matriz de confusión usando solo matplotlib.
    cm        : matriz de confusión (2x2)
    classes   : lista con nombres de clases para los ejes
    filepath  : ruta donde guardar la figura
    normalize : si True, normaliza por fila
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Matriz de Confusión')
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Etiqueta real')
    plt.xlabel('Predicción')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()

# Paso 1 · Cargar y escalar Amount ------------------------------------------
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1); y = df['Class']
X['Amount'] = StandardScaler().fit_transform(X[['Amount']])

# Paso 2 · Split reproducible ------------------------------------------------
split_path='modelos/test_split.pkl'
if os.path.exists(split_path):
    X_train,X_test,y_train,y_test = joblib.load(split_path)
else:
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.30,stratify=y,random_state=42)
    os.makedirs('modelos',exist_ok=True); joblib.dump((X_train,X_test,y_train,y_test),split_path)

# Paso 3 · Modelo logístico (balanceado) ------------------------------------
logreg = LogisticRegression(max_iter=500, solver='lbfgs',
                            class_weight='balanced', n_jobs=-1,
                            C=1.0, penalty='l2', random_state=42)
print("Entrenando Logistic Regression …")
logreg.fit(X_train, y_train)

# Paso 4 · Métricas & gráficas ----------------------------------------------
probas = logreg.predict_proba(X_test)[:,1]; preds=(probas>=0.50).astype(int)
roc_auc = roc_auc_score(y_test, probas); pr_auc=average_precision_score(y_test,probas)
cm = confusion_matrix(y_test,preds)
print(classification_report(y_test,preds,digits=4))
print(f"AUC-ROC={roc_auc:.6f}  AP={pr_auc:.6f}  CM={cm.ravel()}")

os.makedirs('imagenes', exist_ok=True)
# ROC
fpr,tpr,_=roc_curve(y_test,probas); plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
plt.title("ROC – LogReg"); plt.legend([f"AUC={roc_auc:.3f}"]); plt.savefig('imagenes/roc_logreg.png'); plt.close()
# PR
prec,rec,_=precision_recall_curve(y_test,probas); plt.figure(); plt.plot(rec,prec)
plt.title("PR – LogReg"); plt.legend([f"AP={pr_auc:.3f}"]); plt.savefig('imagenes/pr_logreg.png'); plt.close()
# Gain & Confusión
# Curva de Cumulative Gain sin scikit-plot
plot_cumulative_gain(y_test, probas, 'imagenes/gain_logreg.png')

# Matriz de confusión sin scikit-plot
plot_confusion(cm, classes=['No fraude','Fraude'], filepath='imagenes/confmat_logreg.png', normalize=False, cmap='Blues')
# Learning curve
cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
ts,tr,val = learning_curve(logreg,X_train,y_train,cv=cv,scoring='roc_auc',
                            train_sizes=np.linspace(0.1,1.0,6),n_jobs=-1)
plt.figure(); plt.plot(ts,tr.mean(1),'o-',label='Train AUC')
plt.plot(ts,val.mean(1),'o-',label='CV AUC'); plt.legend()
plt.title("Learning Curve – LogReg"); plt.xlabel("Muestras entrenamiento")
plt.savefig('imagenes/learning_curve_logreg.png'); plt.close()

# Paso 5 · Guardar modelo ----------------------------------------------------
joblib.dump(logreg,'modelos/logreg.pkl');  print("✅ Logistic Regression lista.\n")