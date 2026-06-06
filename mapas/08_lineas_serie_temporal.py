# =============================================================================
#  FASE 8 · Serie temporal: evolucion historica de la pobreza multidimensional
#  Macrozona Biobio + Nuble, CASEN 2013 - 2024 (5 olas de la encuesta).
#  Dos series para respetar los cambios metodologicos:
#    * Metodologia 5 dimensiones (oficial vigente, "pobreza_multi_2015"):
#      disponible 2015, 2017, 2022, 2024
#    * Metodologia 4 dimensiones (legacy, "pobreza_multi_4d"):
#      disponible 2013, 2015, 2017, 2022 - puente al pasado pre-2015
#  Salida: Final/serie_temporal_pobreza_multi.png + .html
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

# -- Configuracion por ola CASEN ---------------------------------------------
# Cada entrada describe como cargar y filtrar el archivo de ese ano para que
# la cobertura geografica "Biobio + Nuble" sea homogenea (Nuble nace como
# region 16 recien en sep-2018; antes era parte de la region 8).
OLAS = [
    dict(anio=2013, archivo="casen_2013.dta", jefe="pco1",  regiones=[8]),
    dict(anio=2015, archivo="casen_2015.dta", jefe="pco1",  regiones=[8]),
    dict(anio=2017, archivo="casen_2017.dta", jefe="pco1",  regiones=[8]),
    dict(anio=2022, archivo="casen_2022.dta", jefe="pco1_a", regiones=[8, 16]),
    dict(anio=2024, archivo="casen_2024.dta", jefe="pco1_a", regiones=[8, 16]),
]
DIR_DATOS_HIST = st.DIR_RAIZ / "Datos"

# -- Series a calcular: (nombre, columna CASEN, anios disponibles, color, simbolo)
SERIES = [
    dict(nombre="Metodologia 5 dimensiones<br><sup>(vigente desde 2015)</sup>",
         col="pobreza_multi_2015", anios=[2015, 2017, 2022, 2024],
         color=st.COLOR_MULTI, simbolo="circle", grosor=4.0),
    dict(nombre="Metodologia 4 dimensiones<br><sup>(legacy, hasta 2013)</sup>",
         col="pobreza_multi_4d",   anios=[2013, 2015, 2017, 2022],
         color=st.COLOR_URBANO, simbolo="diamond", grosor=3.0),
]


def media_ponderada(valores, pesos):
    v = pd.to_numeric(valores, errors="coerce")
    p = pd.to_numeric(pesos, errors="coerce")
    mask = v.notna() & p.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=p[mask]))


def cargar_ola(ola):
    """Carga una ola CASEN con las columnas minimas necesarias."""
    archivo = DIR_DATOS_HIST / ola["archivo"]
    # Detectar dinamicamente que columnas estan disponibles
    r = pd.io.stata.StataReader(archivo)
    disponibles = set(r.variable_labels().keys())
    necesarias = {"folio", "region", "expr", ola["jefe"]}
    for s in SERIES:
        if ola["anio"] in s["anios"] and s["col"] in disponibles:
            necesarias.add(s["col"])
    df = pd.read_stata(archivo, convert_categoricals=False,
                       columns=sorted(necesarias))
    df["region"] = pd.to_numeric(df["region"], errors="coerce")
    df = df[df["region"].isin(ola["regiones"])].copy()
    df["expr"] = pd.to_numeric(df["expr"], errors="coerce")
    for s in SERIES:
        if s["col"] in df.columns:
            df[s["col"]] = pd.to_numeric(df[s["col"]], errors="coerce")
    return df


def calcular_serie():
    """Devuelve un DataFrame: anio x columna_serie -> tasa(%) + n personas."""
    filas = []
    for ola in OLAS:
        print(f"  Cargando CASEN {ola['anio']}...")
        df = cargar_ola(ola)
        fila = {"anio": ola["anio"],
                "n_personas": int(len(df)),
                "pob_expandida": float(df["expr"].sum())}
        for s in SERIES:
            if s["col"] in df.columns:
                fila[s["col"]] = media_ponderada(df[s["col"]], df["expr"]) * 100
            else:
                fila[s["col"]] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)


def construir_grafico(serie):
    fig = go.Figure()

    # Banda gris sutil que indica el "puente" 2015-2022 donde las dos
    # metodologias coexisten (verificacion empirica de la brecha 4d vs 5d).
    fig.add_shape(type="rect", xref="x", yref="paper",
                  x0=2014.5, x1=2022.5, y0=0, y1=1,
                  fillcolor="rgba(189, 195, 199, 0.10)", line=dict(width=0),
                  layer="below")
    fig.add_annotation(x=2018.7, y=1.02, xref="x", yref="paper",
                       text="<i>Periodo de coexistencia metodologica (4d y 5d)</i>",
                       showarrow=False, xanchor="center",
                       font=dict(size=10, color="#7F8C8D",
                                 family=st.FONT_FAMILY))

    for s in SERIES:
        sub = serie.dropna(subset=[s["col"]])
        fig.add_trace(go.Scatter(
            x=sub["anio"], y=sub[s["col"]],
            mode="lines+markers+text",
            name=s["nombre"],
            text=[f"<b>{v:.1f}%</b>" for v in sub[s["col"]]],
            textposition="top center",
            textfont=dict(size=11, color=s["color"], family=st.FONT_FAMILY),
            line=dict(color=s["color"], width=s["grosor"], shape="linear"),
            marker=dict(size=11, color=s["color"], symbol=s["simbolo"],
                        line=dict(color="white", width=1.5)),
            customdata=np.stack([sub["n_personas"], sub["pob_expandida"]],
                                axis=-1),
            hovertemplate=("<b>CASEN %{x}</b><br>"
                           "Tasa: %{y:.2f}%<br>"
                           "Muestra: %{customdata[0]:,.0f} personas<br>"
                           "Poblacion expandida: %{customdata[1]:,.0f}"
                           "<extra></extra>"),
        ))

    # Calcular las variaciones acumuladas para anotacion narrativa
    s5d = serie.dropna(subset=["pobreza_multi_2015"])
    s4d = serie.dropna(subset=["pobreza_multi_4d"])
    if len(s5d) >= 2:
        v_ini = s5d["pobreza_multi_2015"].iloc[0]
        v_fin = s5d["pobreza_multi_2015"].iloc[-1]
        delta_5d = v_fin - v_ini
        a_ini = int(s5d["anio"].iloc[0]); a_fin = int(s5d["anio"].iloc[-1])
        signo = "-" if delta_5d < 0 else "+"
        fig.add_annotation(
            x=2024, y=v_fin, xref="x", yref="y",
            text=(f"<b>Caida {a_ini}-{a_fin} (5d)</b><br>"
                  f"{signo}{abs(delta_5d):.1f} pp"),
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=st.COLOR_MULTI, ax=-70, ay=-50,
            xanchor="right",
            font=dict(family=st.FONT_FAMILY, size=10.5,
                      color=st.COLOR_TEXTO),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=st.COLOR_MULTI, borderwidth=1, borderpad=5,
        )

    # Anotacion explicativa del salto metodologico 4d -> 5d
    if not s5d.empty and not s4d.empty:
        comunes = sorted(set(s5d["anio"]) & set(s4d["anio"]))
        if comunes:
            a_ref = comunes[0]
            v5 = s5d.loc[s5d["anio"] == a_ref, "pobreza_multi_2015"].iloc[0]
            v4 = s4d.loc[s4d["anio"] == a_ref, "pobreza_multi_4d"].iloc[0]
            brecha = v5 - v4
            fig.add_annotation(
                x=a_ref, y=(v5 + v4) / 2, xref="x", yref="y",
                text=(f"<b>+{brecha:.1f} pp</b><br>"
                      "<sup>aporte de la 5° dim.<br>(Redes y cohesion social)</sup>"),
                showarrow=False, xanchor="left", xshift=15,
                font=dict(family=st.FONT_FAMILY, size=10,
                          color=st.COLOR_TEXTO),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#D5DBDB", borderwidth=1, borderpad=5,
            )

    # Nota Nuble (region 16 desde 2018)
    fig.add_annotation(
        x=2017.5, y=-0.16, xref="x", yref="paper",
        text=("<i>Nota: en 2013-2017 Nuble era parte administrativa de la "
              "region del Biobio (region 8);<br>"
              "en 2022-2024 ya figura como region 16. La cobertura geografica "
              "se mantiene homogenea filtrando ambos casos.</i>"),
        showarrow=False, xanchor="center",
        font=dict(size=10, color="#7F8C8D", family=st.FONT_FAMILY),
    )

    y_max = max(serie[[s["col"] for s in SERIES]].max().max() * 1.18, 30)

    fig.update_layout(
        title=dict(
            text=("<b>Una decada de pobreza multidimensional en Biobio + Nuble: "
                  "caida sostenida, pero piso aun alto</b><br>"
                  "<sup>Tasa de personas en hogares con pobreza multidimensional "
                  "segun ambas metodologias oficiales. Macrozona "
                  "Biobio + Nuble, ponderado por factor regional (expr) - "
                  "CASEN 2013, 2015, 2017, 2022 y 2024.</sup>"),
            font=dict(family=st.FONT_FAMILY, size=17,
                      color=st.COLOR_TEXTO_FUERTE),
            x=0.02, xanchor="left", pad=dict(t=15, b=15),
        ),
        xaxis=dict(
            title=dict(text="Ano de la encuesta CASEN",
                       font=dict(size=12)),
            tickmode="array",
            tickvals=[o["anio"] for o in OLAS],
            ticktext=[str(o["anio"]) for o in OLAS],
            range=[2012, 2025.2],
            gridcolor="#ECF0F1", zeroline=False,
            showline=True, linecolor="#DFE6E9", tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text="% de la poblacion en pobreza multidimensional",
                       font=dict(size=12)),
            ticksuffix="%", range=[0, y_max],
            gridcolor="#ECF0F1", zeroline=False, tickfont=dict(size=11),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1.0,
                    font=dict(size=11, family=st.FONT_FAMILY),
                    bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family=st.FONT_FAMILY, color=st.COLOR_TEXTO),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=140, b=140, l=80, r=80),
        width=1150, height=650,
        annotations=list(fig.layout.annotations) + [dict(
            x=0.02, y=-0.32, xref="paper", yref="paper",
            text=("<i>Fuente: Encuestas CASEN 2013, 2015, 2017, 2022 y 2024 "
                  "(MDSF) - elaboracion propia. La 5° dimension (Redes y "
                  "cohesion social) se incorpora en 2015.</i>"),
            showarrow=False, xanchor="left",
            font=dict(size=9.5, color="#7F8C8D"),
        )],
    )
    return fig


def main():
    print("FASE 8 | Construyendo serie temporal Biobio + Nuble (2013-2024)...")
    serie = calcular_serie()

    print("\n  Tasas (%) por ola CASEN:")
    cols_show = ["anio", "n_personas"] + [s["col"] for s in SERIES]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(serie[cols_show].round(2).to_string(index=False))

    fig = construir_grafico(serie)
    png_out = st.DIR_FINAL / "serie_temporal_pobreza_multi.png"
    html_out = st.DIR_FINAL / "serie_temporal_pobreza_multi.html"
    fig.write_image(png_out, width=1150, height=650, scale=2.5)
    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"\n  Guardado: {png_out.name}")
    print(f"  Guardado: {html_out.name}")
    print("FASE 8 completada.\n")


if __name__ == "__main__":
    main()
