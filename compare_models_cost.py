#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nombre: compare_models_cost.py
Autor: Marcos Ibáñez Fandos
Fecha: 23/07/2025
Universidad: Universitat Politècnica de València (UPV)
Proyecto TFG: Optimización de costes en detección de fraude

Descripción:
  Este script compara cinco modelos de detección de fraude (RandomForest Baseline, RandomForest Optimizado,
  XGBoost, LightGBM y Regresión Logística), evaluando su rendimiento según su coste total bajo distintas
  restricciones operativas y regulatorias.

Objetivo:
  Determinar el umbral de decisión óptimo que minimice el coste total (FN + FP), respetando:
    • Recall mínimo global.
    • Recall mínimo en fraudes de alto importe (≥ €10.000).
    • Restricciones normativas que exigen revisión manual para transacciones ≥ €10.000.

Estructura de costes:
  • FN (Fraudes no detectados): penalizados según su importe y gravedad (micro, medio, alto),
    aplicando markups diferenciados según percentiles P33/P66.
  • FP (Falsos positivos):
      - Automáticos: importe ≤ €100, procesados sin intervención humana.
      - Manuales: requieren revisión por analistas humanos.

Lógica de revisión:
  - El umbral T_low (=100 €) se aplica primero: todo FP con importe ≤ T_low se resuelve automáticamente.
  - Si el importe ≥ €10.000, la revisión es obligatoriamente manual (CDD rule).
  - Para el resto de FP de importe intermedio, se asignan primero a revisión manual hasta agotar la capacidad,
    priorizando los de mayor importe (los de menor importe se van a revisión automática si hay overflow).

Modos de ejecución:
  • Normativa:
      - Aplica restricciones de regulación europeas.
      - Simula una empresa mediana con 20 analistas, trabajando 16 h y revisando 40 casos/h.
      - El script calcula la capacidad máxima de revisiones manuales y asigna el resto como FP automáticos.

  • Personalizado:
      - Permite definir todos los parámetros: costes, markups, restricciones, número de analistas y tiempo operativo.
      - El usuario también puede nombrar la simulación.

Salidas generadas:
  • CSV de métricas y costes: `metrics_coste_<escenario>.csv`
  • Matrices de confusión por modelo: `confmat_coste_<escenario>.png`
  • Barras de coste total: `coste_barras_<escenario>.png`
  • Barras de métricas: `metrics_barras_<escenario>.png`
  • Curvas Precision-Recall y ROC: `pr_<escenario>.png`, `roc_<escenario>.png`
  • Todos los ficheros se guardan en la carpeta `comparar_coste/`

Uso:
  • Interactivo: pregunta parámetros al usuario
      python compare_models_cost.py
  • Por argumentos:
      python compare_models_cost.py --mode normativa
      python compare_models_cost.py --mode personalizado
"""

import os, argparse, sys, joblib
import re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import InconsistentVersionWarning
# Silenciar todas las InconsistentVersionWarning de sklearn al deserializar modelos
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", module="sklearn.base")
warnings.filterwarnings("ignore")

# --- Silenciar logs y warnings de LightGBM ---
import logging
# Configurar logging de LightGBM para solo mostrar errores
logging.getLogger('lightgbm').setLevel(logging.ERROR)
# Silenciar advertencias específicas de LightGBM sobre "best gain: -inf"
warnings.filterwarnings("ignore", message="No further splits with positive gain")
# Ignorar cualquier advertencia procedente del módulo lightgbm
warnings.filterwarnings("ignore", module="lightgbm")
import lightgbm as lgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score
)


# Cargar media y desviación de "Amount" para convertir z-scores a euros en leyendas
AMT_MEAN = AMT_STD = None
def _load_amount_stats(csv_path="creditcard.csv"):
    global AMT_MEAN, AMT_STD
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, usecols=["Amount"])
            AMT_MEAN = df["Amount"].mean()
            AMT_STD  = df["Amount"].std(ddof=0)
        except: pass

def z_to_eur(z):
    if AMT_MEAN is None or AMT_STD is None:
        return z
    return z * AMT_STD + AMT_MEAN
# -------------------------------------------------------------------------

# Parche de compatibilidad imblearn/sklearn v1.7+ para evitar errores de tags
try:
    from sklearn.utils._tags import _safe_tags
except:
    import sklearn.utils._tags as _t
    def _safe_tags(est, key=None, default=None):
        tags = getattr(est, "_get_tags", lambda: {})()
        return tags.get(key, default) if key else tags
    _t._safe_tags = _safe_tags
# -------------------------------------------------------------------------


def buscar_umbral_coste_optimo(y_true, scores, amounts,
                               C_FN, C_FP_auto, C_FP_manual, T_low,
                               min_rec_high=0.0, min_rec_all=0.0, min_prec_all=0.0, high_amt_never_pass=0.0,
                               markup_micro=1.1, markup_med=1.2, markup_alto=1.5):
    
# -----------------------------------------------------------------------------
# Función central del análisis: busca el umbral óptimo de clasificación para 
# cada modelo minimizando el coste total, teniendo en cuenta costes de FP y FN,
# restricciones normativas y recall mínimo en fraudes de alto importe. 
# Esta función aplica la lógica de priorización y segmentación definida por la normativa.
# -----------------------------------------------------------------------------

    # -------------------- Validación cruzada cost-sensitive (scaffold) --------------------
    # TODO: Integrar búsqueda cost-sensitive por validación cruzada (KFold, n_splits=3) y promediar costes.
    # Por ahora, se mantiene la búsqueda original pero dejamos el scaffold para futura implementación.
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    best_costs = {}
    # Estructura para futura integración:
    # for train_idx, val_idx in kf.split(scores):
    #     # Para cada fold, llamar recursivamente a la misma función sin CV (helper interno _buscar_umbral_fold)
    #     pass  # placeholder

    # -------------------- Búsqueda de umbral cost-sensitive (lógica existente) --------------------
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts)
    pos = (y_true == 1)
    neg = ~pos

    p33, p66 = np.percentile(amounts[pos], [33, 66])
    uniq = np.unique(scores)
    thr_cand = uniq.tolist()

    all_costs, all_thr = [], []
    best_cost = np.inf
    best_thr, best_rec, best_f1 = 0.5, -1, -1
    best_counts = (0, 0, 0)
    best_fn_amount = 0

    low_mask = (amounts < T_low)
    total = len(thr_cand)
    step = max(1, total // 10)
    print_indices = set(list(range(0, total, step)) + [total - 1])
    print(f"   Buscando coste óptimo entre {len(thr_cand)} umbrales...", flush=True)

    PENALIZADOR_RECALL = 2.0  # Penalización proporcional si no se alcanza el recall mínimo

    for i, t in enumerate(thr_cand):
        y_pred = scores >= t
        fpm = neg & y_pred
        fnm = pos & ~y_pred

        rec_all_i = ((scores >= t) & pos).sum() / pos.sum() if pos.sum() > 0 else 0
        tp_temp = (pos & (scores >= t)).sum()
        fp_temp = fpm.sum()
        prec_all_i = tp_temp / (tp_temp + fp_temp) if (tp_temp + fp_temp) > 0 else 0

        # Se elimina el "continue" para no descartar ningún umbral
        # y en su lugar se aplica penalización más fuerte
        if min_rec_all > 0 and rec_all_i < min_rec_all:
            recall_deficit = min_rec_all - rec_all_i
        else:
            recall_deficit = 0

        # Clasificación FP automáticos y manuales
        fp_auto = fpm & low_mask
        fp_man = fpm & ~fp_auto

        # Regla CDD: revisión manual obligatoria ≥ 10.000 €
        cdd_mask = amounts >= 10000.0
        fp_man = fp_man | (fpm & cdd_mask)
        fp_auto = fp_auto & ~cdd_mask

        n_fn = fnm.sum()
        amount_fn = amounts[fnm].sum()

        fn_amounts = amounts[fnm]
        sum_micro = fn_amounts[fn_amounts <= p33].sum()
        sum_med = fn_amounts[(fn_amounts > p33) & (fn_amounts <= p66)].sum()
        sum_alto = fn_amounts[fn_amounts > p66].sum()

        cost_fn = C_FN * (
            markup_micro * sum_micro +
            markup_med * sum_med +
            markup_alto * sum_alto
        )

        # Penalización proporcional si no cumple el recall mínimo
        if recall_deficit > 0:
            cost_fn *= (1 + PENALIZADOR_RECALL * recall_deficit)

        cost_fp = fp_auto.sum() * C_FP_auto + fp_man.sum() * C_FP_manual
        total_cost = cost_fn + cost_fp

        all_costs.append(total_cost)
        all_thr.append(t)

        tp = (pos & y_pred).sum()
        prec = tp / (tp + fp_auto.sum() + fp_man.sum()) if tp else 0
        rec = tp / (tp + n_fn) if tp else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        if (total_cost < best_cost or
            (total_cost == best_cost and rec > best_rec) or
            (total_cost == best_cost and rec == best_rec and f1 > best_f1)):
            best_cost, best_thr, best_rec, best_f1, best_counts = total_cost, t, rec, f1, (
                fp_auto.sum(), fp_man.sum(), n_fn)
            best_fn_amount = amount_fn

        if i in print_indices:
            print(f"   · {i+1}/{total} umbrales | best_cost={int(best_cost)}€ @ thr={best_thr:.4f}", flush=True)

    # Fallback en caso de no encontrar nada válido
    if not np.isfinite(best_cost) or best_cost == np.inf:
        idx_best = int(np.argmin(all_costs))
        best_cost = all_costs[idx_best]
        best_thr = all_thr[idx_best]

    safe_cost = int(np.round(best_cost)) if np.isfinite(best_cost) else 0
    return best_thr, safe_cost, best_counts, best_fn_amount

def segment_fn_amounts(amounts, fn_mask, p33, p66, C_FN_micro, C_FN_medio, C_FN_alto):
    """Segmenta FN en 3 grupos por importe y calcula costes."""
    fn_amts = amounts[fn_mask]
    n1 = np.sum(fn_amts<=p33)
    n2 = np.sum((fn_amts>p33)&(fn_amts<=p66))
    n3 = np.sum(fn_amts>p66)
    return (int(n1),int(n2),int(n3),
            int(n1*C_FN_micro), int(n2*C_FN_medio), int(n3*C_FN_alto))


# --- Funciones de plot con caja de leyenda baja central ---------------

def _draw_legend(text):
    # Dividir el texto por ';', eliminar espacios y limitar a 3 líneas
    lines = text.split(';')
    lines = [ln.strip() for ln in lines]
    if len(lines) > 3:
        lines = [lines[0], '; '.join(lines[1:-1]), lines[-1]]
    text_block = "\n".join(lines)
    plt.gcf().text(
        0.5, 0.05, text_block,
        ha='center', va='bottom',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    )

def plot_confmatrix_grid(confs, title, path, legend_text=None):
    n = len(confs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    # Dibujar los valores de cada celda en la matriz de confusión
    for ax, (cm, thr, name, cost) in zip(axes, confs):
        im = ax.imshow(cm, cmap="Blues")
        for i in (0, 1):
            for j in (0, 1):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)
        ax.set_title(f"{name}\nthr={thr:.3f}\ncoste={cost}€", fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.30, 1, 0.95])
    plt.subplots_adjust(bottom=0.35)
    if legend_text:
        _draw_legend(legend_text)
    fig.savefig(path)
    plt.close(fig)

def plot_barras_coste(df, title, path, legend_text=None):
    plt.figure(figsize=(10,6))
    # Resaltar modelos que no cumplen las restricciones de umbral
    if 'ok_thr' in df.columns:
        colors = ['C0' if ok else 'red' for ok in df['ok_thr']]
    else:
        colors = ['C0'] * len(df)
    bars = plt.bar(df['modelo'], df['coste'], color=colors)
    ymax=df['coste'].max()
    for b,v in zip(bars, df['coste']):
        plt.text(b.get_x()+b.get_width()/2, v+0.02*ymax,
                 f"{v}€", ha='center', va='bottom', fontsize=9)
    plt.title(title); plt.ylabel("Coste total (€)")
    plt.xticks(rotation=15)
    plt.tight_layout(rect=[0,0.18,1,0.95])
    plt.subplots_adjust(bottom=0.25)
    if legend_text: _draw_legend(legend_text)
    plt.savefig(path); plt.close()

def plot_barras_metricas(df, title, path, legend_text=None):
    plt.figure(figsize=(10,6))
    idx = np.arange(len(df)); w = 0.25
    # Usar colores fijos para cada métrica: verde para Precisión, naranja para Recall, azul para F1
    # No es necesario calcular colors_ok para este gráfico
    bars_prec = plt.bar(idx-w, df['precision'], w, label='Prec', color='green')
    bars_rec  = plt.bar(idx,   df['recall'],    w, label='Rec', color='orange')
    bars_f1   = plt.bar(idx+w, df['f1'],        w, label='F1', color='blue')
    plt.xticks(idx, df['modelo'], rotation=15)
    plt.ylim(0,1.05)
    # Anotar cada barra con su valor numérico
    for bar in bars_prec:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in bars_rec:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    for bar in bars_f1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha='center', va='bottom', fontsize=8)
    plt.title(title)
    plt.legend()
    plt.tight_layout(rect=[0,0.18,1,0.95])
    plt.subplots_adjust(bottom=0.25)
    if legend_text:
        _draw_legend(legend_text)
    plt.savefig(path)
    plt.close()

def plot_pr_curves(curves, apd, title, path, legend_text=None, thr_points=None):
    plt.figure(figsize=(8,6))
    for nm,(p,r) in curves.items():
        plt.plot(r,p,label=f"{nm} (AP={apd[nm]:.3f})")
    if thr_points:
        for nm, (rec_thr, prec_thr) in thr_points.items():
            plt.scatter(rec_thr, prec_thr, marker='o', s=50, edgecolor='black')
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend(loc='lower left'); plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0,0.22,1,0.95])
    plt.subplots_adjust(bottom=0.30)
    if legend_text: _draw_legend(legend_text)
    plt.savefig(path); plt.close()

def plot_roc_curves(curves, aud, title, path, legend_text=None, thr_points=None):
    plt.figure(figsize=(8,6))
    for nm,(fpr,tpr) in curves.items():
        plt.plot(fpr,tpr,label=f"{nm} (AUC={aud[nm]:.3f})")
    plt.plot([0,1],[0,1],'k--',alpha=0.4)
    if thr_points:
        for nm, (fpr_thr, tpr_thr) in thr_points.items():
            plt.scatter(fpr_thr, tpr_thr, marker='o', s=50, edgecolor='black')
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title); plt.legend(loc='lower right'); plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0,0.22,1,0.95])
    plt.subplots_adjust(bottom=0.30)
    if legend_text: _draw_legend(legend_text)
    plt.savefig(path); plt.close()

# ------------------------------- MAIN -----------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", dest="mode", type=str, default="normativa",
                   help="Modo de ejecución: normativa o personalizado")
    p.add_argument("--C_FN", type=float, default=3.64)
    p.add_argument("--C_FP_auto", type=float, default=0.04)
    p.add_argument("--C_FP_manual", type=float, default=4.0)
    p.add_argument("--T_low", type=float, default=100.0)
    p.add_argument("--markup_micro", type=float, default=7.5)
    p.add_argument("--markup_med", type=float, default=3.0)
    p.add_argument("--markup_alto", type=float, default=1.5)
    p.add_argument("--min_rec_high", type=float, default=0.99)
    p.add_argument("--min_rec_all", type=float, default=0.90)
    p.add_argument("--min_prec_all", type=float, default=0.0)
    p.add_argument("--high_amt_never_pass", type=float, default=10000.0)
    p.add_argument("--nombre", type=str, default="normativa")
    p.add_argument("--revisiones_por_hora", type=float, default=40.0, help="Revisiones por hora por analista")
    p.add_argument("--horas_test", type=float, default=16, help="Horas totales de test")
    p.add_argument("--analistas_disponibles", type=int, default=20, help="Número de analistas disponibles")
   
    args = p.parse_args()

   

    # Asignación de parámetros
    C_FN = args.C_FN
    C_FP_auto = args.C_FP_auto
    C_FP_manual = args.C_FP_manual
    T_low = args.T_low
    markup_micro = args.markup_micro
    markup_med = args.markup_med
    markup_alto = args.markup_alto
    min_rec_high = args.min_rec_high
    min_rec_all = args.min_rec_all
    min_prec_all = args.min_prec_all
    high_amt_never_pass = args.high_amt_never_pass
    escenario = args.nombre
    mode = args.mode.strip().lower()

    if mode not in ['normativa', 'personalizado']:
        print("⚠️ Modo no reconocido. Usa 'normativa' o 'personalizado'.")
        sys.exit(1)

    # --- Normativa ---
    if mode == 'normativa':
        print("🧩 Ejecutando en modo normativa")
        revisiones_por_hora = 40
        horas_test = 16
        analistas_disponibles = 20
        capacidad_manual = int(revisiones_por_hora * horas_test * analistas_disponibles)

    # --- Personalizado ---
    elif mode == 'personalizado':
        print("⚙️ Ejecutando en modo personalizado")
        revisiones_por_hora = args.revisiones_por_hora
        horas_test = args.horas_test
        analistas_disponibles = args.analistas_disponibles
        capacidad_manual = int(revisiones_por_hora * horas_test * analistas_disponibles)

    # Asegurar datos cargados
    _load_amount_stats("creditcard.csv")
    os.makedirs("comparar_coste", exist_ok=True)

    # Resto del código principal del script (idéntico)
    # -------------------------------------------------
    data = joblib.load("modelos/test_split.pkl")
    if isinstance(data, tuple):
        if len(data) == 2:
            X_train, y_train = None, None
            X_test, y_test = data
        elif len(data) == 4:
            X_train, X_test, y_train, y_test = data
        else:
            raise RuntimeError("Formato inesperado de test_split.pkl")
    else:
        raise RuntimeError("Formato inesperado de test_split.pkl")
    if X_train is None or y_train is None:
        raise RuntimeError("No hay datos de entrenamiento para calibración")

    if 'Amount' not in X_test.columns:
        raise RuntimeError("X_test debe tener columna 'Amount'")
    amounts = z_to_eur(X_test['Amount'].values)

    p33, p66 = np.percentile(amounts[y_test == 1], [33, 66])
    if 'Amount' not in X_train.columns:
        raise RuntimeError("X_train debe tener columna 'Amount'")
    amounts_train = z_to_eur(X_train['Amount'].values)

    modelos = {
        "RF Baseline": "modelos/rf_baseline.pkl",
        "RF Optimizado": "modelos/rf_optimized.pkl",
        "XGBoost": "modelos/xgboost.pkl",
        "LightGBM": "modelos/lightgbm.pkl",
        "LogisticReg": "modelos/logreg.pkl",
    }

    fallback_models = []
    legend_txt = (
        f"C_FP_auto={C_FP_auto}€; C_FP_manual={C_FP_manual}€\n"
        f"min_rec_high={min_rec_high}; min_rec_all={min_rec_all}; min_prec_all={min_prec_all}\n"
        f"high_amt_never_pass={high_amt_never_pass:.2f}€"
    )

    thr_pr_points, thr_roc_points, rows = {}, {}, []
    pr_curves, roc_curves, apd, aud, confs = {}, {}, {}, {}, []

    for name, path in modelos.items():
        print(f"→ {name}")
        raw_model = joblib.load(path)
        if hasattr(raw_model, 'named_steps') and 'clf' in raw_model.named_steps:
            clf = raw_model.named_steps['clf']
        else:
            clf = raw_model

        from lightgbm import LGBMClassifier
        if isinstance(clf, LGBMClassifier):
            clf.set_params(verbosity=-1, verbose=-1)

        scores_test = clf.predict_proba(X_test)[:, 1]

        thr, cost, (fa, fm, fn), sum_fn_amt = buscar_umbral_coste_optimo(
            y_test, scores_test, amounts, C_FN, C_FP_auto, C_FP_manual, T_low,
            min_rec_high=min_rec_high,
            min_rec_all=min_rec_all,
            min_prec_all=min_prec_all,
            high_amt_never_pass=high_amt_never_pass,
            markup_micro=markup_micro,
            markup_med=markup_med,
            markup_alto=markup_alto
        )

        y_pred = (scores_test >= thr).astype(int)
        fn_mask_test = (y_test == 1) & (y_pred == 0)
        neg_mask_test = (y_test == 0)
        fp_mask_test = neg_mask_test & (y_pred == 1)

        low_mask_test = (amounts < T_low)
        cdd_mask_test = amounts >= 10000.0
        fp_low = fp_mask_test & low_mask_test & ~cdd_mask_test
        fp_high = fp_mask_test & ~low_mask_test | (fp_mask_test & cdd_mask_test)

        if mode == 'normativa':
            indices_fp_high = np.where(fp_high)[0]
            sorted_fp_high_idx = indices_fp_high[np.argsort(amounts[indices_fp_high])]
            if len(sorted_fp_high_idx) > capacidad_manual:
                idx_manual = sorted_fp_high_idx[-capacidad_manual:]
                idx_auto = sorted_fp_high_idx[:-capacidad_manual]
            else:
                idx_manual = sorted_fp_high_idx
                idx_auto = np.array([], dtype=int)
            fp_man_mask_test = np.zeros_like(fp_high, dtype=bool)
            fp_auto_mask_test = np.zeros_like(fp_low, dtype=bool)
            fp_man_mask_test[idx_manual] = True
            fp_auto_mask_test[idx_auto] = True
            fp_auto_mask_test |= fp_low
        else:
            indices_fp_high = np.where(fp_high)[0]
            sorted_fp_high_idx = indices_fp_high[np.argsort(amounts[indices_fp_high])]
            if len(sorted_fp_high_idx) > capacidad_manual:
                idx_manual = sorted_fp_high_idx[-capacidad_manual:]
                idx_auto = sorted_fp_high_idx[:-capacidad_manual]
            else:
                idx_manual = sorted_fp_high_idx
                idx_auto = np.array([], dtype=int)
            fp_man_mask_test = np.zeros_like(fp_high, dtype=bool)
            fp_auto_mask_test = np.zeros_like(fp_low, dtype=bool)
            fp_man_mask_test[idx_manual] = True
            fp_auto_mask_test[idx_auto] = True
            fp_auto_mask_test |= fp_low

        fn_test_amounts = amounts[fn_mask_test]
        sum_micro = fn_test_amounts[fn_test_amounts <= p33].sum()
        sum_med = fn_test_amounts[(fn_test_amounts > p33) & (fn_test_amounts <= p66)].sum()
        sum_alto = fn_test_amounts[fn_test_amounts > p66].sum()
        cost_fn_test = C_FN * (markup_micro * sum_micro + markup_med * sum_med + markup_alto * sum_alto)
        cost_fp_test = fp_auto_mask_test.sum() * C_FP_auto + fp_man_mask_test.sum() * C_FP_manual
        cost_total = cost_fn_test + cost_fp_test

        fa, fm, fn = fp_auto_mask_test.sum(), fp_man_mask_test.sum(), fn_mask_test.sum()
        sum_fn_amt = amounts[fn_mask_test].sum()
        tp = ((y_test == 1) & (y_pred == 1)).sum()
        tn = ((y_test == 0) & (y_pred == 0)).sum()
        prec = tp / (tp + fa + fm) if tp else 0
        rec     = tp / (tp + fn) if tp else 0
        f1v = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        ap = average_precision_score(y_test, scores_test)
        fpr, tpr, _ = roc_curve(y_test, scores_test)
        ru = auc(fpr, tpr)
        pr, rc, _ = precision_recall_curve(y_test, scores_test)
        cm = confusion_matrix(y_test, y_pred)

        # Marcar como inválido si recall=1 con threshold=0 y sin coste de fraude
        modelo_valido = not (
            rec == 1.0 and thr == 0 and fn == 0 and
            sum_fn_amt == 0 and cost_fn_test == 0
        )

        rows.append({
            'modelo': name, 'precision': prec, 'recall': rec, 'f1': f1v,
            'ap': ap, 'aucroc': ru, 'thr': thr,
            'fp_auto': fa, 'fp_manual': fm, 'fn': fn, 'tp': tp, 'tn': tn,
            'fn_amount_sum': int(sum_fn_amt),
            'cost_fn': int(cost_fn_test), 'cost_fp_auto': fa * C_FP_auto,
            'cost_fp_manual': fm * C_FP_manual, 'coste': int(cost_total),
            'ok_thr': modelo_valido
        })

        pr_curves[name] = (pr, rc)
        roc_curves[name] = (fpr, tpr)
        apd[name] = ap
        aud[name] = ru
        confs.append((cm, thr, name, int(cost_total)))

        print(f"   umbral={thr:.3f}   coste={int(cost_total)}€   FP_auto={fa}   FP_man={fm}   FN={fn}")

    df = pd.DataFrame(rows).sort_values('coste')
    df.to_csv(f"comparar_coste/metrics_coste_{escenario}.csv", index=False)
    print(df.to_string(index=False))

    plot_confmatrix_grid(confs, f"Matrices de confusión (modo={escenario})",
                         f"comparar_coste/confmat_coste_{escenario}.png", legend_txt)
    plot_barras_coste(df[['modelo', 'coste', 'ok_thr']], f"Coste total (modo={escenario})",
                      f"comparar_coste/coste_barras_{escenario}.png", legend_txt)
    plot_barras_metricas(df[['modelo', 'precision', 'recall', 'f1', 'ok_thr']],
                         f"Precision/Recall/F1 (modo={escenario})",
                         f"comparar_coste/metrics_barras_{escenario}.png", legend_txt)
    plot_pr_curves(pr_curves, apd, f"Precision-Recall (modo={escenario})",
                   f"comparar_coste/pr_{escenario}.png", legend_txt, thr_points=thr_pr_points)
    plot_roc_curves(roc_curves, aud, f"ROC (modo={escenario})",
                    f"comparar_coste/roc_{escenario}.png", legend_txt, thr_points=thr_roc_points)

    print("✔ Hecho. Ficheros en comparar_coste/")

if __name__ == "__main__":
    main()