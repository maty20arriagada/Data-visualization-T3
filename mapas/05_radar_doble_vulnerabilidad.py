# =============================================================================
#  FASE 5 · Radar plot: doble vulnerabilidad multidimensional
#  Compara las 3 provincias identificadas en el mapa bivariado (Arauco, Punilla,
#  Itata - bajo ingreso + alta dependencia de subsidios) contra las 3 restantes
#  (Concepcion, Biobio, Diguillin) sobre las 6 dimensiones canonicas de la
#  pobreza multidimensional CASEN. Cierra la historia narrativa de los mapas.
#  Salida: Final/radar_doble_vulnerabilidad.png + .html
# =============================================================================
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

# --- Definicion de grupos basada en el mapa bivariado (mapa_3) -------------
PROV_DOBLE_VULN = {82, 162, 163}   # Arauco, Punilla, Itata
PROV_RESTO = {81, 83, 161}         # Concepcion, Biobio, Diguillin

# --- 6 dimensiones de la pobreza multidimensional CASEN ---------------------
DIMENSIONES = [
    ("Hacinamiento<br>(Vivienda)",          "hh_d_hacina_2015"),
    ("Entorno precario<br>(Vivienda)",      "hh_d_entorno_2015"),
    ("Falta de escolaridad<br>(Educacion)", "hh_d_esc"),
    ("Falta de atencion<br>(Salud)",        "hh_d_acc"),
    ("Trabajo informal<br>(Seg. Social)",   "hh_d_contprev"),
    ("Desocupacion<br>(Trabajo)",           "hh_d_actsub"),
]

GRUPOS = [
    ("Doble vulnerabilidad", PROV_DOBLE_VULN, st.COLOR_MULTI,
     "rgba(231, 76, 60, 0.25)", "square",
     ["Arauco", "Punilla", "Itata"]),
    ("Resto de la macrozona", PROV_RESTO, st.COLOR_URBANO,
     "rgba(43, 91, 132, 0.32)", "circle",
     ["Concepcion", "Biobio", "Diguillin"]),
]

# Posicion de la etiqueta numerica en cada vertice, alineada radialmente hacia
# afuera segun el angulo del eje (con rotation=90 y direccion clockwise).
TEXTPOS_POR_VERTICE = [
    "top center",    # 0  Hacinamiento (arriba)
    "top right",     # 1  Entorno (arriba-derecha)
    "bottom right",  # 2  Falta escolaridad (abajo-derecha)
    "bottom center", # 3  Falta atencion salud (abajo)
    "bottom left",   # 4  Trabajo informal (abajo-izquierda)
    "top left",      # 5  Desocupacion (arriba-izquierda)
]
# Umbral: si la brecha entre los dos grupos en un eje es menor a este valor (pp),
# se oculta la etiqueta numerica del grupo "Resto" para evitar superposicion con
# la del grupo "Doble vulnerabilidad" (los puntos quedan visualmente juntos).
UMBRAL_BRECHA_PP = 1.5


def media_ponderada(valores, pesos):
    """Mismo helper que en 02_etl_indicadores.py — media ponderada robusta."""
    v = pd.to_numeric(valores, errors="coerce")
    p = pd.to_numeric(pesos, errors="coerce")
    mask = v.notna() & p.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=p[mask]))


def cargar_datos():
    print("FASE 5 | Radar 'Doble vulnerabilidad': cargando CASEN 2024...")
    cols = ["folio", "id_persona", "region", "pco1_a"] + \
           [c for _, c in DIMENSIONES]
    principal = pd.read_stata(st.CASEN_PRINCIPAL, convert_categoricals=False,
                              columns=cols)
    prov_comuna = pd.read_stata(st.CASEN_PROV_COMUNA, convert_categoricals=False)
    df = principal.merge(prov_comuna, on=["folio", "id_persona"], how="left")
    df = df[df["region"].isin(st.CODIGOS_REGION)].copy()
    for _, c in DIMENSIONES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["expp"] = pd.to_numeric(df["expp"], errors="coerce")
    return df


def calcular_tasas(df):
    """Devuelve dict {grupo: {'tasas': [6], 'n_pers':int, 'n_hog':int, 'pob':float}}."""
    resultados = {}
    for nombre, codigos, *_ in GRUPOS:
        sub = df[df["provincia"].isin(codigos)]
        tasas = [media_ponderada(sub[col], sub["expp"]) * 100
                 for _, col in DIMENSIONES]
        resultados[nombre] = {
            "tasas": tasas,
            "n_pers": int(len(sub)),
            "n_hog": int((sub["pco1_a"] == 1).sum()),
            "pob": float(sub["expp"].sum()),
        }
    return resultados


def construir_radar(resultados):
    ejes = [n for n, _ in DIMENSIONES]
    ejes_cerrado = ejes + [ejes[0]]

    # Escala radial: justo al siguiente multiplo de 10 (sin holgura excesiva)
    max_global = max(max(d["tasas"]) for d in resultados.values())
    escala_max = math.ceil(max_global / 10) * 10

    # Brechas por eje (para ocultar etiquetas redundantes en "Resto")
    dv = resultados["Doble vulnerabilidad"]["tasas"]
    rs = resultados["Resto de la macrozona"]["tasas"]
    ocultar_resto = [abs(d - r) < UMBRAL_BRECHA_PP for d, r in zip(dv, rs)]

    textpos_cerrado = TEXTPOS_POR_VERTICE + [TEXTPOS_POR_VERTICE[0]]

    fig = go.Figure()
    for (nombre, _, color_linea, color_relleno, simbolo, provs) in GRUPOS:
        d = resultados[nombre]
        tasas_cerradas = d["tasas"] + [d["tasas"][0]]
        if nombre == "Resto de la macrozona":
            textos = [("" if oculta else f"{v:.1f}%")
                      for v, oculta in zip(d["tasas"], ocultar_resto)]
        else:
            textos = [f"{v:.1f}%" for v in d["tasas"]]
        textos = textos + [textos[0]]

        fig.add_trace(go.Scatterpolar(
            r=tasas_cerradas, theta=ejes_cerrado,
            mode="lines+markers+text",
            text=textos, textposition=textpos_cerrado,
            textfont=dict(size=10.5, color=color_linea, family=st.FONT_FAMILY),
            marker=dict(size=9, color=color_linea, symbol=simbolo,
                        line=dict(color="white", width=1.2)),
            fill="toself", fillcolor=color_relleno,
            line=dict(color=color_linea, width=3.5),
            name=(f"<b>{nombre}</b><br>"
                  f"{', '.join(provs)}<br>"
                  f"<i>n = {d['n_pers']:,} personas "
                  f"({d['n_hog']:,} hogares)</i>"),
            hovertemplate=("<b>%{theta}</b><br>" + nombre +
                           ": %{r:.2f}%<extra></extra>"),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#FAFAFA",
            radialaxis=dict(
                visible=True, range=[0, escala_max], dtick=10,
                ticksuffix="%", angle=30, tickangle=0,
                tickfont=dict(size=11, color="#34495E", family=st.FONT_FAMILY),
                gridcolor="#E5E7E9", linecolor="#E5E7E9",
            ),
            angularaxis=dict(
                tickfont=dict(size=12.5, color=st.COLOR_TEXTO,
                              family=st.FONT_FAMILY),
                linecolor="#BDC3C7", gridcolor="#E5E7E9",
                rotation=90, direction="clockwise",
            ),
        ),
        title=dict(
            text=("<b>Vulnerabilidad con cara propia: educacion y entorno "
                  "marcan la brecha</b><br>"
                  "<sup>Las provincias del mapa bivariado (Arauco, Punilla, "
                  "Itata) no acumulan mas carencias en todas las dimensiones; "
                  "su rezago se vuelca en escolaridad y entorno,<br>mientras "
                  "que en hacinamiento - tipico de zonas urbanas densas - estan "
                  "incluso en mejor pie que el resto de la macrozona. "
                  "CASEN 2024, ponderado por expp.</sup>"),
            font=dict(size=17, color=st.COLOR_TEXTO_FUERTE,
                      family=st.FONT_FAMILY),
            x=0.5, xanchor="center", pad=dict(t=15, b=20),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22,
            xanchor="center", x=0.5,
            font=dict(size=11.5, family=st.FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.92)",
        ),
        font=dict(family=st.FONT_FAMILY, color=st.COLOR_TEXTO),
        margin=dict(t=160, b=170, l=130, r=130),
        height=820, width=1020,
        paper_bgcolor="white", plot_bgcolor="white",
        annotations=[dict(
            x=0.0, y=-0.32, xref="paper", yref="paper",
            text=f"<i>{st.FUENTE_NOTA}</i>",
            showarrow=False, font=dict(size=9.5, color="#7F8C8D"),
            xanchor="left",
        )],
    )
    return fig


def main():
    df = cargar_datos()
    resultados = calcular_tasas(df)

    # Reporte por consola + validacion de la hipotesis
    print("\n  Tasas ponderadas por grupo (% poblacion afectada):")
    print(f"  {'Dimension':<35} {'Doble vuln.':>12} {'Resto':>10} "
          f"{'Brecha':>10}")
    print("  " + "-" * 70)
    dv = resultados["Doble vulnerabilidad"]["tasas"]
    rs = resultados["Resto de la macrozona"]["tasas"]
    brechas = []
    for (etiqueta, _), v_dv, v_rs in zip(DIMENSIONES, dv, rs):
        brecha = v_dv - v_rs
        brechas.append(brecha)
        print(f"  {etiqueta.replace('<br>', ' '):<35} "
              f"{v_dv:>11.1f}% {v_rs:>9.1f}% {brecha:>+9.1f} pp")
    print("  " + "-" * 70)
    for grupo in ["Doble vulnerabilidad", "Resto de la macrozona"]:
        d = resultados[grupo]
        print(f"  {grupo}: n = {d['n_pers']:,} personas "
              f"({d['n_hog']:,} hogares, pob. expandida = {d['pob']:,.0f})")
    dims_donde_dv_es_peor = sum(1 for b in brechas if b > 0)
    print(f"\n  Hipotesis: 'Doble vulnerab.' > 'Resto' en >=5/6 ejes -> "
          f"se cumple en {dims_donde_dv_es_peor}/6")

    fig = construir_radar(resultados)
    png_out = st.DIR_FINAL / "radar_doble_vulnerabilidad.png"
    html_out = st.DIR_FINAL / "radar_doble_vulnerabilidad.html"
    fig.write_image(png_out, width=1020, height=820, scale=2.5)
    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"\n  Guardado: {png_out.name}")
    print(f"  Guardado: {html_out.name}")
    print("FASE 5 completada.\n")


if __name__ == "__main__":
    main()
