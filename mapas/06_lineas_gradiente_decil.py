# =============================================================================
#  FASE 6 · Grafico de lineas: gradiente socioeconomico por decil de ingreso
#  Muestra como descienden 4 dimensiones de la pobreza multidimensional a
#  medida que sube el ingreso autonomo per capita del hogar, dentro de la
#  macrozona Biobio + Nuble (CASEN 2024). Cierra la pregunta que dejaron
#  abiertos los mapas y el radar: ¿que tan ligadas al ingreso son las
#  carencias estructurales?
#  Salida: Final/lineas_gradiente_decil.png + .html
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

# (etiqueta legible, columna CASEN, color, simbolo del marker)
INDICADORES = [
    ("Pobreza multidimensional",   "pobreza_multi",     st.COLOR_MULTI,     "circle"),
    ("Falta de escolaridad",       "hh_d_esc",          st.COLOR_RURAL,     "diamond"),
    ("Carencia acceso a salud",    "hh_d_acc",          st.COLOR_URBANO,    "square"),
    ("Hacinamiento",               "hh_d_hacina_2015",  "#8E44AD",          "triangle-up"),
]


def cuantiles_ponderados(valores, pesos, k=10):
    """Devuelve los k-1 puntos de corte para dividir en k grupos de peso
    aproximadamente igual (cuantiles ponderados)."""
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


def cargar_y_clasificar():
    print("FASE 6 | Cargando CASEN 2024 y clasificando en deciles ponderados...")
    cols = (["folio", "id_persona", "region", "pco1_a", "numper", "yautcorh"]
            + [c for _, c, *_ in INDICADORES])
    principal = pd.read_stata(st.CASEN_PRINCIPAL, convert_categoricals=False,
                              columns=cols)
    prov_comuna = pd.read_stata(st.CASEN_PROV_COMUNA, convert_categoricals=False)
    df = principal.merge(prov_comuna, on=["folio", "id_persona"], how="left")
    df = df[df["region"].isin(st.CODIGOS_REGION)].copy()

    for c in ["yautcorh", "numper", "expp"] + [c for _, c, *_ in INDICADORES]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ingreso autonomo per capita del hogar (constante dentro de cada folio)
    df["ing_autonomo_pc"] = np.where(df["numper"] > 0,
                                     df["yautcorh"] / df["numper"], np.nan)

    # Deciles a NIVEL HOGAR (un registro por folio) ponderados por expp
    hog = df[df["pco1_a"] == 1].drop_duplicates("folio").copy()
    hog = hog.dropna(subset=["ing_autonomo_pc", "expp"])
    cortes = cuantiles_ponderados(hog["ing_autonomo_pc"], hog["expp"], k=10)
    bins = [-np.inf] + cortes + [np.inf]
    hog["decil"] = pd.cut(hog["ing_autonomo_pc"], bins=bins,
                          labels=list(range(1, 11)), include_lowest=True
                          ).astype(int)
    print(f"  Hogares clasificados: {len(hog):,} | "
          f"Puntos de corte (CLP/mes): " +
          ", ".join(f"D{i+1}<{c:,.0f}" for i, c in enumerate(cortes)))

    # Propagar el decil a todas las personas del hogar
    df = df.merge(hog[["folio", "decil"]], on="folio", how="left")
    df = df.dropna(subset=["decil"])
    df["decil"] = df["decil"].astype(int)
    return df, hog


def calcular_curvas(df, hog):
    """Devuelve un DataFrame con columnas: decil, ingreso_medio,
    poblacion, n_personas, y una columna por indicador (% personas)."""
    filas = []
    for d in range(1, 11):
        sub = df[df["decil"] == d]
        sub_hog = hog[hog["decil"] == d]
        fila = {
            "decil": d,
            "ingreso_medio": media_ponderada(sub_hog["ing_autonomo_pc"],
                                             sub_hog["expp"]),
            "poblacion": float(sub["expp"].sum()),
            "n_personas": int(len(sub)),
        }
        for etiqueta, col, *_ in INDICADORES:
            fila[etiqueta] = media_ponderada(sub[col], sub["expp"]) * 100
        filas.append(fila)
    curvas = pd.DataFrame(filas)
    return curvas


def construir_grafico(curvas):
    fig = go.Figure()

    # 4 lineas, una por indicador
    for etiqueta, col, color, simbolo in INDICADORES:
        valores = curvas[etiqueta]
        ancho = 4.5 if col == "pobreza_multi" else 3
        fig.add_trace(go.Scatter(
            x=curvas["decil"], y=valores, mode="lines+markers",
            name=etiqueta,
            line=dict(color=color, width=ancho, shape="linear"),
            marker=dict(symbol=simbolo, size=10, color=color,
                        line=dict(color="white", width=1.5)),
            customdata=np.stack([curvas["ingreso_medio"],
                                 curvas["poblacion"]], axis=-1),
            hovertemplate=(f"<b>{etiqueta}</b><br>"
                           "Decil %{x} (ingreso medio $%{customdata[0]:,.0f})<br>"
                           "Tasa: %{y:.1f}%<br>"
                           "Poblacion del decil: %{customdata[1]:,.0f}"
                           "<extra></extra>"),
        ))
        # Etiquetas numericas en los extremos (decil 1 y decil 10)
        for i_extremo, anchor in [(0, "right"), (9, "left")]:
            v = valores.iloc[i_extremo]
            fig.add_annotation(
                x=curvas["decil"].iloc[i_extremo], y=v,
                text=f"<b>{v:.1f}%</b>",
                xanchor=anchor, yanchor="middle",
                xshift=-9 if anchor == "right" else 9,
                showarrow=False,
                font=dict(family=st.FONT_FAMILY, size=10.5, color=color),
            )

    # Banda con el ingreso medio en deciles clave (1, 5, 10) bajo el eje X
    bandas = [
        (1,  f"D1: ingreso medio<br>${curvas['ingreso_medio'].iloc[0]:,.0f}"),
        (5,  f"D5: ${curvas['ingreso_medio'].iloc[4]:,.0f}"),
        (10, f"D10: ${curvas['ingreso_medio'].iloc[9]:,.0f}"),
    ]
    for d, txt in bandas:
        fig.add_annotation(
            x=d, y=-0.18, xref="x", yref="paper",
            text=f"<i>{txt}</i>", showarrow=False,
            font=dict(family=st.FONT_FAMILY, size=9.5, color="#7F8C8D"),
            xanchor="center",
        )

    # Brecha decil 1 vs decil 10 (anotacion narrativa)
    pm = curvas["Pobreza multidimensional"]
    brecha_pm = pm.iloc[0] - pm.iloc[9]
    fe = curvas["Falta de escolaridad"]
    brecha_fe = fe.iloc[0] - fe.iloc[9]
    fig.add_annotation(
        x=10.4, y=pm.iloc[9], xref="x", yref="y",
        text=(f"<b>Brecha D1–D10</b><br>"
              f"Pobreza multi: <b>-{brecha_pm:.0f} pp</b><br>"
              f"Escolaridad: <b>-{brecha_fe:.0f} pp</b>"),
        showarrow=False, xanchor="left", yanchor="middle",
        font=dict(family=st.FONT_FAMILY, size=10, color=st.COLOR_TEXTO),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#D5DBDB", borderwidth=1, borderpad=6,
    )

    fig.update_layout(
        title=dict(
            text=("<b>El gradiente socioeconomico: la pobreza no cae al mismo "
                  "ritmo en todas sus dimensiones</b><br>"
                  "<sup>Tasa de cada carencia (% poblacion afectada) segun "
                  "decil de ingreso autonomo per capita del hogar. "
                  "Macrozona Biobio + Nuble, ponderado por expp - CASEN 2024."
                  "</sup>"),
            font=dict(family=st.FONT_FAMILY, size=17,
                      color=st.COLOR_TEXTO_FUERTE),
            x=0.02, xanchor="left", pad=dict(t=15, b=15),
        ),
        xaxis=dict(
            title=dict(text="Decil de ingreso autonomo per capita "
                            "(1 = 10% mas pobre  -  10 = 10% mas rico)",
                       font=dict(size=12)),
            tickmode="array", tickvals=list(range(1, 11)),
            ticktext=[f"D{i}" for i in range(1, 11)],
            range=[0.5, 11.6], gridcolor="#ECF0F1", zeroline=False,
            showline=True, linecolor="#DFE6E9", tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text="% de la poblacion afectada",
                       font=dict(size=12)),
            ticksuffix="%", range=[0, max(curvas[[e for e, *_ in INDICADORES]]
                                          .max()) * 1.18],
            gridcolor="#ECF0F1", zeroline=False, tickfont=dict(size=11),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="right", x=1.0,
                    font=dict(size=11, family=st.FONT_FAMILY),
                    bgcolor="rgba(255,255,255,0.85)"),
        font=dict(family=st.FONT_FAMILY, color=st.COLOR_TEXTO),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=130, b=120, l=80, r=100),
        width=1150, height=620,
        annotations=list(fig.layout.annotations) + [dict(
            x=0.02, y=-0.32, xref="paper", yref="paper",
            text=f"<i>{st.FUENTE_NOTA}</i>",
            showarrow=False, xanchor="left",
            font=dict(size=9.5, color="#7F8C8D"),
        )],
    )
    return fig


def main():
    df, hog = cargar_y_clasificar()
    curvas = calcular_curvas(df, hog)

    print("\n  Tasas (%) por decil:")
    cols_show = ["decil", "ingreso_medio"] + [e for e, *_ in INDICADORES]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(curvas[cols_show].round(1).to_string(index=False))

    fig = construir_grafico(curvas)
    png_out = st.DIR_FINAL / "lineas_gradiente_decil.png"
    html_out = st.DIR_FINAL / "lineas_gradiente_decil.html"
    fig.write_image(png_out, width=1150, height=620, scale=2.5)
    fig.write_html(html_out, include_plotlyjs="cdn")
    print(f"\n  Guardado: {png_out.name}")
    print(f"  Guardado: {html_out.name}")
    print("FASE 6 completada.\n")


if __name__ == "__main__":
    main()
