# =============================================================================
#  FASE 7 · Gradientes facetados (small multiples)
#  4 paneles con la MISMA bateria de carencias multidimensionales, contrastadas
#  con 4 lentes distintas: nivel socioeconomico (decil de ingreso), ciclo de
#  vida (edad del jefe de hogar), tamano del hogar y capital humano (anios de
#  escolaridad del jefe). Cierra la pregunta "¿cuales son los predictores mas
#  fuertes de cada carencia?" - macrozona Biobio + Nuble.
#  Salida: Final/lineas_gradiente_multifacet.png + .html
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

# (etiqueta, columna CASEN, color, simbolo, grosor de linea)
INDICADORES = [
    ("Pobreza multidimensional",  "pobreza_multi",     st.COLOR_MULTI,  "circle",       4.5),
    ("Falta de escolaridad",      "hh_d_esc",          st.COLOR_RURAL,  "diamond",      3),
    ("Carencia acceso a salud",   "hh_d_acc",          st.COLOR_URBANO, "square",       3),
    ("Hacinamiento",              "hh_d_hacina_2015",  "#8E44AD",       "triangle-up",  3),
]

# -- Definicion de los 4 lentes (paneles) ------------------------------------
# Cada panel: (titulo, columna fuente en df, funcion clasificadora -> Serie de
# categoria 1..K, etiquetas del eje X, descripcion para hover)
def _bins_etario(s):
    bins = [-np.inf, 29, 44, 59, 74, np.inf]
    return pd.cut(s, bins=bins, labels=[1, 2, 3, 4, 5]).astype(float)

def _bins_tamano(s):
    bins = [-np.inf, 1, 2, 3, 4, np.inf]
    return pd.cut(s, bins=bins, labels=[1, 2, 3, 4, 5]).astype(float)

def _bins_esc(s):
    # 0 sin esc, 1-8 basica, 9-11 media incompleta, 12 media completa, 13+ sup.
    bins = [-np.inf, 0, 8, 11, 12, np.inf]
    return pd.cut(s, bins=bins, labels=[1, 2, 3, 4, 5]).astype(float)

PANELES = [
    # Panel 1 (especial): deciles de ingreso autonomo p/c, ponderados
    dict(titulo="a. Por decil de ingreso autonomo per capita del hogar",
         x_label="Decil  (D1 = 10% mas pobre ··· D10 = 10% mas rico)",
         tickvals=list(range(1, 11)),
         ticktext=[f"D{i}" for i in range(1, 11)],
         tipo="decil"),
    # Panel 2: edad del jefe de hogar
    dict(titulo="b. Por ciclo de vida (edad del jefe de hogar)",
         x_label="Tramo etario del jefe de hogar",
         tickvals=[1, 2, 3, 4, 5],
         ticktext=["18-29", "30-44", "45-59", "60-74", "75+"],
         tipo="categoria", fuente="edad_jefe", clasif=_bins_etario),
    # Panel 3: tamano del hogar
    dict(titulo="c. Por tamano del hogar (numero de personas)",
         x_label="Numero de personas en el hogar",
         tickvals=[1, 2, 3, 4, 5],
         ticktext=["1", "2", "3", "4", "5+"],
         tipo="categoria", fuente="numper_jefe", clasif=_bins_tamano),
    # Panel 4: escolaridad del jefe
    dict(titulo="d. Por capital humano (anios de escolaridad del jefe)",
         x_label="Nivel educacional alcanzado por el jefe de hogar",
         tickvals=[1, 2, 3, 4, 5],
         ticktext=["Sin esc.<br>(0 anios)", "Basica<br>(1-8)",
                   "Media inc.<br>(9-11)", "Media comp.<br>(12)",
                   "Superior<br>(13+)"],
         tipo="categoria", fuente="esc_jefe", clasif=_bins_esc),
]


# ---- Helpers ---------------------------------------------------------------
def cuantiles_ponderados(valores, pesos, k=10):
    df = pd.DataFrame({"v": valores, "w": pesos}).dropna().sort_values("v")
    cw = df["w"].cumsum().values
    total = cw[-1]
    return [df["v"].iloc[int(np.searchsorted(cw, total * i / k))]
            for i in range(1, k)]


def media_ponderada(valores, pesos):
    v = pd.to_numeric(valores, errors="coerce")
    p = pd.to_numeric(pesos, errors="coerce")
    mask = v.notna() & p.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=p[mask]))


# ---- ETL -------------------------------------------------------------------
def cargar():
    print("FASE 7 | Cargando CASEN 2024 (macrozona Biobio + Nuble)...")
    cols = (["folio", "id_persona", "region", "pco1_a", "numper", "edad",
             "esc", "yautcorh"] + [c for _, c, *_ in INDICADORES])
    principal = pd.read_stata(st.CASEN_PRINCIPAL, convert_categoricals=False,
                              columns=cols)
    prov_comuna = pd.read_stata(st.CASEN_PROV_COMUNA, convert_categoricals=False)
    df = principal.merge(prov_comuna, on=["folio", "id_persona"], how="left")
    df = df[df["region"].isin(st.CODIGOS_REGION)].copy()

    for c in (["yautcorh", "numper", "expp", "edad", "esc"]
              + [c for _, c, *_ in INDICADORES]):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ing_autonomo_pc"] = np.where(df["numper"] > 0,
                                     df["yautcorh"] / df["numper"], np.nan)

    # Atributos del jefe replicados en todas las personas del hogar
    jefes = (df[df["pco1_a"] == 1][["folio", "edad", "esc", "numper"]]
             .drop_duplicates("folio")
             .rename(columns={"edad": "edad_jefe", "esc": "esc_jefe",
                              "numper": "numper_jefe"}))
    df = df.merge(jefes, on="folio", how="left")

    # Decil de ingreso (a nivel HOGAR ponderado por expp), propagado a personas
    hog = (df[df["pco1_a"] == 1].drop_duplicates("folio")
           .dropna(subset=["ing_autonomo_pc", "expp"]))
    cortes = cuantiles_ponderados(hog["ing_autonomo_pc"], hog["expp"], k=10)
    bins = [-np.inf] + cortes + [np.inf]
    hog = hog.assign(decil=pd.cut(hog["ing_autonomo_pc"], bins=bins,
                                  labels=list(range(1, 11)),
                                  include_lowest=True).astype(int))
    df = df.merge(hog[["folio", "decil"]], on="folio", how="left")
    df["decil"] = df["decil"].astype(float)
    return df, hog


# ---- Calculo de curvas por panel -------------------------------------------
def curvas_panel(df, hog, panel):
    if panel["tipo"] == "decil":
        grupos = sorted(df["decil"].dropna().unique().astype(int))
        col_grupo = "decil"
    else:
        df = df.copy()
        df["_grupo"] = panel["clasif"](df[panel["fuente"]])
        grupos = sorted(df["_grupo"].dropna().unique().astype(int))
        col_grupo = "_grupo"

    filas = []
    for g in grupos:
        sub = df[df[col_grupo] == g]
        fila = {"grupo": int(g),
                "etiqueta": panel["ticktext"][int(g) - 1],
                "poblacion": float(sub["expp"].sum()),
                "n_personas": int(len(sub))}
        for etiqueta, col, *_ in INDICADORES:
            fila[etiqueta] = media_ponderada(sub[col], sub["expp"]) * 100
        filas.append(fila)
    return pd.DataFrame(filas)


# ---- Grafico ---------------------------------------------------------------
def construir_grafico(curvas_por_panel):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[p["titulo"] for p in PANELES],
        shared_yaxes=True, vertical_spacing=0.32, horizontal_spacing=0.08,
    )
    posiciones = {(1, 1): 0, (1, 2): 1, (2, 1): 2, (2, 2): 3}
    y_global_max = max(
        max(curvas[[e for e, *_ in INDICADORES]].max()) for curvas in curvas_por_panel
    )

    for (r, c), idx in posiciones.items():
        panel = PANELES[idx]
        curvas = curvas_por_panel[idx]
        mostrar_leyenda = (r == 1 and c == 1)
        for etiqueta, col_v, color, simbolo, grosor in INDICADORES:
            fig.add_trace(go.Scatter(
                x=curvas["grupo"], y=curvas[etiqueta], mode="lines+markers",
                name=etiqueta, legendgroup=etiqueta,
                showlegend=mostrar_leyenda,
                line=dict(color=color, width=grosor, shape="linear"),
                marker=dict(symbol=simbolo, size=9, color=color,
                            line=dict(color="white", width=1.2)),
                customdata=np.stack(
                    [curvas["etiqueta"], curvas["poblacion"]], axis=-1),
                hovertemplate=(f"<b>{etiqueta}</b><br>"
                               f"{panel['x_label'].split('(')[0].strip()}: "
                               "%{customdata[0]}<br>"
                               "Tasa: %{y:.1f}%<br>"
                               "Poblacion del grupo: %{customdata[1]:,.0f}"
                               "<extra></extra>"),
            ), row=r, col=c)

        # Eje X
        fig.update_xaxes(
            tickmode="array", tickvals=panel["tickvals"],
            ticktext=panel["ticktext"], title_text=panel["x_label"],
            title_font=dict(size=11), title_standoff=18,
            tickfont=dict(size=10),
            gridcolor="#ECF0F1", zeroline=False, showline=True,
            linecolor="#DFE6E9", row=r, col=c,
        )
        # Eje Y (solo titulo en columna izquierda)
        fig.update_yaxes(
            range=[0, y_global_max * 1.10], ticksuffix="%",
            gridcolor="#ECF0F1", zeroline=False, tickfont=dict(size=10),
            title_text=("% poblacion afectada" if c == 1 else None),
            title_font=dict(size=11), row=r, col=c,
        )

    # Subtitulo de cada subplot mas chico y alineado a izquierda
    for ann in fig.layout.annotations[:4]:
        ann.font = dict(family=st.FONT_FAMILY, size=12.5,
                        color=st.COLOR_TEXTO_FUERTE)
        ann.x = ann.x
        ann.xanchor = "center"

    fig.update_layout(
        title=dict(
            text=("<b>4 lentes sobre la misma vulnerabilidad: ingreso, edad, "
                  "tamano del hogar y educacion del jefe</b><br>"
                  "<sup>Como varia cada dimension de la pobreza multidimensional "
                  "segun cuatro predictores estructurales. Macrozona Biobio + "
                  "Nuble, ponderado por expp - CASEN 2024.</sup>"),
            font=dict(family=st.FONT_FAMILY, size=18,
                      color=st.COLOR_TEXTO_FUERTE),
            x=0.02, xanchor="left", pad=dict(t=18, b=15),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                    font=dict(size=11.5, family=st.FONT_FAMILY),
                    bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family=st.FONT_FAMILY, color=st.COLOR_TEXTO),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=170, b=90, l=85, r=40),
        width=1350, height=1000,
        annotations=list(fig.layout.annotations) + [dict(
            x=0.02, y=-0.07, xref="paper", yref="paper",
            text=f"<i>{st.FUENTE_NOTA}</i>",
            showarrow=False, xanchor="left",
            font=dict(size=10, color="#7F8C8D"),
        )],
    )
    return fig


def main():
    df, hog = cargar()
    curvas_por_panel = [curvas_panel(df, hog, p) for p in PANELES]

    print("\n  Resumen de tasas por panel:")
    for panel, curvas in zip(PANELES, curvas_por_panel):
        print(f"\n  --- {panel['titulo']} ---")
        cols_show = ["etiqueta", "n_personas"] + [e for e, *_ in INDICADORES]
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(curvas[cols_show].round(1).to_string(index=False))

    fig = construir_grafico(curvas_por_panel)
    png_out = st.DIR_FINAL / "lineas_gradiente_multifacet.png"
    html_out = st.DIR_FINAL / "lineas_gradiente_multifacet.html"
    fig.write_image(png_out, width=1350, height=1000, scale=2.5)
    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"\n  Guardado: {png_out.name}")
    print(f"  Guardado: {html_out.name}")
    print("FASE 7 completada.\n")


if __name__ == "__main__":
    main()
