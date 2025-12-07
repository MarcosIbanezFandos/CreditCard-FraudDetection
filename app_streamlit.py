import streamlit as st
import subprocess
import os
import pandas as pd
import altair as alt
import json
import io
import matplotlib.pyplot as plt
import zipfile
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
import time
import datetime

st.set_page_config(page_title="Simulador de Costes de Modelos de Fraude", layout="wide")

if "ejecuto_simulacion" not in st.session_state:
    st.session_state.ejecuto_simulacion = False

st.title("Simulador de Estrategias de Detección de Fraude")

tabs = st.tabs(["⚙️ Configuración", "📊 Resultados", "📥 Exportar"])

with tabs[0]:
    st.header("⚙️ Parámetros de Simulación")

    preset = st.radio(
        "🎯 ¿Qué tipo de simulación deseas realizar?",
        options=["Normativa", "Personalizado"],
        index=0,
        horizontal=True,
        help="Selecciona 'Normativa' para usar los parámetros estándar o 'Personalizado' para ajustarlos tú mismo."
    )

    presets_dict = {
        "Normativa": {
            "C_FN": 3.64, "C_FP_auto": 0.04, "C_FP_manual": 4.0, "T_low": 100.0,
            "markup_micro": 7.5, "markup_med": 3.0, "markup_alto": 1.5,
            "min_rec_high": 0.99, "min_rec_all": 0.90, "min_prec_all": 0.0,
            "high_amt_never_pass": 10000.0, "nombre": "normativa",
            "revisiones_por_hora": 40.0, "horas_test": 16.0, "analistas_disponibles": 20
        },
        "Personalizado": {}
    }

    params = presets_dict.get(preset, {})

    if preset == "Personalizado":
        st.subheader("📋 Parámetros Personalizados")
        with st.expander("💸 Costes"):
            params["C_FN"] = st.number_input("Coste por euro de fraude no detectado", 0.1, 10.0, 3.64)
            params["C_FP_auto"] = st.number_input("Coste por FP automático", 0.0, 5.0, 0.04)
            params["C_FP_manual"] = st.number_input("Coste por FP manual", 0.0, 100.0, 4.0)
            params["T_low"] = st.number_input("Umbral T_low para FP automáticos", 0.0, 5000.0, 100.0)

        with st.expander("📈 Markup e Indicadores"):
            params["markup_micro"] = st.slider("Markup micro (≤ P33)", 1.0, 20.0, 7.5)
            params["markup_med"] = st.slider("Markup medio", 1.0, 10.0, 3.0)
            params["markup_alto"] = st.slider("Markup alto", 1.0, 5.0, 1.5)
            params["min_rec_high"] = st.slider("Recall mínimo ≥10.000€", 0.0, 1.0, 0.99)
            params["min_rec_all"] = st.slider("Recall mínimo global", 0.0, 1.0, 0.90)
            params["min_prec_all"] = st.slider("Precisión mínima", 0.0, 1.0, 0.0)
            params["high_amt_never_pass"] = st.number_input("Importe alto no permitido", 0.0, 50000.0, 10000.0)

        with st.expander("👥 Capacidad manual"):
            params["revisiones_por_hora"] = st.number_input("Revisiones/hora/analista", 0.0, 200.0, 40.0)
            params["horas_test"] = st.number_input("Horas de test", 0.0, 48.0, 16.0)
            params["analistas_disponibles"] = st.number_input("Nº de analistas", 0, 200, 20)

        params["nombre"] = st.text_input("Nombre de simulación", "personalizado")

    ejecutar = st.button("🚀 Ejecutar Comparación de Modelos")

    if ejecutar:
        st.session_state.ejecuto_simulacion = True

        tiempo_estimado = 124
        status_placeholder = st.empty()
        status_placeholder.info("⏳ Ejecutando simulación...")
        progress_bar = st.progress(0)

        cmd = [
            "python3", "compare_models_cost.py",
            "--mode", preset.lower(),
            "--C_FN", str(params["C_FN"]),
            "--C_FP_auto", str(params["C_FP_auto"]),
            "--C_FP_manual", str(params["C_FP_manual"]),
            "--T_low", str(params["T_low"]),
            "--markup_micro", str(params["markup_micro"]),
            "--markup_med", str(params["markup_med"]),
            "--markup_alto", str(params["markup_alto"]),
            "--min_rec_high", str(params["min_rec_high"]),
            "--min_rec_all", str(params["min_rec_all"]),
            "--min_prec_all", str(params["min_prec_all"]),
            "--high_amt_never_pass", str(params["high_amt_never_pass"]),
            "--nombre", str(params["nombre"]),
            "--revisiones_por_hora", str(params["revisiones_por_hora"]),
            "--horas_test", str(params["horas_test"]),
            "--analistas_disponibles", str(params["analistas_disponibles"])
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / tiempo_estimado)
            progress_bar.progress(progress)
            if process.poll() is not None:
                break
            time.sleep(0.2)

        stdout, stderr = process.communicate()
        progress_bar.progress(1.0)
        progress_bar.empty()
        status_placeholder.empty()

        st.success(f"✅ Simulación completada.")
        st.text_area("🖥️ Salida del script", stdout)
        if stderr:
            st.error("⚠️ Error:" + stderr)
with tabs[1]:
    st.header("📊 Resultados")

    if not st.session_state.ejecuto_simulacion:
        st.info("ℹ️ Selecciona unos parámetros en la pestaña de Configuración y ejecuta la simulación para ver aquí los resultados.")
    else:
        df_path = f"comparar_coste/metrics_coste_{params['nombre']}.csv"
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)

            modelo_menor_coste = df["coste"].idxmin()

            def highlight_rows(row):
                if row.name == modelo_menor_coste:
                    return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
                elif 'ok_thr' in row and not row['ok_thr']:
                    return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                return [''] * len(row)

            st.subheader("📋 Métricas clave", help="🟩 Verde = modelo más barato, 🟥 Rojo = inválido por recall=1.")
            st.dataframe(df.style.apply(highlight_rows, axis=1), use_container_width=True)

            modelo_top = df[df["ok_thr"]].sort_values("coste").iloc[0]
            st.success(f"🏆 Modelo ganador: **{modelo_top['modelo']}** – Coste total estimado: **{int(modelo_top['coste'])} €**")

            st.subheader("💰 Costes por tipo")
            chart_data = df.set_index("modelo")[["cost_fn", "cost_fp_auto", "cost_fp_manual"]]
            st.bar_chart(chart_data)

            st.subheader("📉 Gráfico Coste vs Recall")

            ordenado = df.sort_values("coste").copy()

            # Colores predeterminados por modelo
            colores_base = [
                "#1f77b4",  # azul
                "#ff7f0e",  # naranja
                "#2ca02c",  # verde
                "#d62728",  # rojo
                "#9467bd",  # morado
                "#8c564b",  # marrón
                "#e377c2",  # rosa
                "#7f7f7f",  # gris
                "#bcbd22",  # oliva
                "#17becf"   # cian
            ]

            # Mapear modelos a colores únicos
            modelos = ordenado["modelo"].tolist()
            colores_personalizados = {modelo: colores_base[i % len(colores_base)] for i, modelo in enumerate(modelos)}

            # Sobrescribir en verde si es el más eficiente
            colores_personalizados[modelo_top["modelo"]] = "green"

            # Rojo para los que no cumplen ok_thr
            for idx, row in ordenado.iterrows():
                if not row["ok_thr"]:
                    colores_personalizados[row["modelo"]] = "red"

            ordenado["leyenda_color"] = ordenado["modelo"]

            # Gráfico con color por modelo y leyenda
            scatter = alt.Chart(ordenado).mark_circle(size=120).encode(
                x=alt.X("recall", title="Recall"),
                y=alt.Y("coste", title="Coste (€)"),
                color=alt.Color("leyenda_color:N", legend=alt.Legend(title="Modelo"), scale=alt.Scale(domain=list(colores_personalizados.keys()), range=list(colores_personalizados.values()))),
                tooltip=["modelo", "coste", "recall", "precision", "f1"]
            ).properties(title="Coste vs Recall", width=750, height=400)

            st.altair_chart(scatter, use_container_width=True)

            st.subheader("🧠 Análisis")
            for _, row in df.iterrows():
                comentario = f"- **{row['modelo']}** → coste: **{row['coste']:.0f} €**."
                if not row['ok_thr']:
                    comentario += " ❌ Invalidado por `recall=1` (sin FN detectables)."
                elif row.name == modelo_menor_coste:
                    comentario += " ✅ Más eficiente."
                st.markdown(comentario)
        else:
            st.warning("⚠️ No se encontraron resultados. Ejecuta una simulación primero.")

with tabs[2]:
    st.header("📥 Exportar")

    if "df" in locals() and not df.empty:
        # Exportar como ZIP
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"params_{params['nombre']}.json", json.dumps(params, indent=4))
            zip_file.writestr(f"resultados_{params['nombre']}.csv", df.to_csv(index=False))
        zip_buffer.seek(0)

        st.download_button(
            label="📦 Descargar .zip (Parámetros + Resultados)",
            data=zip_buffer,
            file_name=f"simulacion_{params['nombre']}.zip",
            mime="application/zip"
        )

        # Exportar como PDF
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        # 🏷️ Título del informe
        elements.append(Paragraph("Informe de Simulación de Modelos de Fraude", styles["Heading1"]))
        elements.append(Spacer(1, 0.25*inch))

        # 📋 Parámetros usados
        elements.append(Paragraph("<b>📋 Parámetros de la simulación</b>", styles["Heading2"]))
        for key, value in params.items():
            elements.append(Paragraph(f"{key}: <b>{value}</b>", styles["Normal"]))
        elements.append(Spacer(1, 0.25*inch))

        # 📊 Resultados de los modelos (toda la tabla)
        elements.append(Paragraph("<b>📊 Resultados de los modelos evaluados</b>", styles["Heading2"]))
        table_data = [list(df.columns)] + df.values.tolist()

        # Formatear números grandes y floats
        formatted_table = []
        for row in table_data:
            formatted_row = []
            for item in row:
                if isinstance(item, (int, float)):
                    formatted_row.append(f"{item:,.4f}" if abs(item) < 1000 else f"{item:,.0f}")
                else:
                    formatted_row.append(str(item))
            formatted_table.append(formatted_row)

        # --- Crear tabla PDF con ajuste automático al ancho de la página ---
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib.pagesizes import A4

        # Calcular el ancho disponible (restando márgenes)
        page_width, page_height = A4
        usable_width = page_width - 80  # margen lateral

        # Definir anchos proporcionales por columna
        num_cols = len(formatted_table[0])
        col_widths = [usable_width / num_cols] * num_cols

        # Crear tabla
        table = Table(formatted_table, colWidths=col_widths, repeatRows=1)

        # Estilo de tabla optimizado
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),  # encabezados más pequeños
            ('FONTSIZE', (0, 1), (-1, -1), 6),  # cuerpo más pequeño
            ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ])

        table.setStyle(table_style)
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        # 🧠 Análisis interpretativo
        elements.append(Paragraph("<b>🧠 Análisis interpretativo</b>", styles["Heading2"]))
        for _, row in df.iterrows():
            comentario = f"- <b>{row['modelo']}</b>: coste total <b>{row['coste']:.0f} €</b>."
            if not row["ok_thr"]:
                comentario += " <font color='red'>❌ Recall = 1 (inválido)</font>"
            elif row.name == df["coste"].idxmin():
                comentario += " <font color='green'>✅ Modelo más eficiente (menor coste)</font>"
            elements.append(Paragraph(comentario, styles["Normal"]))
        elements.append(Spacer(1, 0.3*inch))

        # 🏁 Conclusión general
        best_model = df.sort_values("coste").iloc[0]["modelo"]
        best_cost = int(df.sort_values("coste").iloc[0]["coste"])
        elements.append(Paragraph(
            f"<b>🏁 Conclusión:</b> El modelo <b>{best_model}</b> presenta el menor coste total estimado ({best_cost} €), "
            f"cumpliendo los umbrales de calidad definidos en la simulación. Los demás modelos muestran mayores costes o incumplen condiciones de recall.",
            styles["Normal"]
        ))

        # Generar el PDF
        doc.build(elements)
        pdf_buffer.seek(0)

        st.download_button(
            label="📝 Descargar resumen completo en PDF",
            data=pdf_buffer,
            file_name=f"resumen_completo_{params['nombre']}.pdf",
            mime="application/pdf"
        )