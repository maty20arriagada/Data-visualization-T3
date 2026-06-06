# =============================================================================
#  FASE 4 · Mapas interactivos (Plotly)
#  Versiones interactivas (HTML, hover por provincia) de los mapas estaticos,
#  consumiendo el mismo GeoJSON (Fase 1) e indicadores (Fase 2).
#  Salida: Final/mapa_*_interactivo.html
# =============================================================================
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

ESCALA_SECUENCIAL = [
    [0.0, st.ESCALA_SECUENCIAL[0]],
    [0.33, st.ESCALA_SECUENCIAL[1]],
    [0.66, st.ESCALA_SECUENCIAL[2]],
    [1.0, st.ESCALA_SECUENCIAL[3]],
]
ESCALA_DIVERGENTE = [
    [0.0, st.ESCALA_DIVERGENTE[0]],
    [0.25, st.ESCALA_DIVERGENTE[1]],
    [0.5, st.ESCALA_DIVERGENTE[2]],
    [0.75, st.ESCALA_DIVERGENTE[3]],
    [1.0, st.ESCALA_DIVERGENTE[4]],
]
ESCALA_SALUD = [[0.0, "#EAF1F6"], [0.33, "#9DC3D9"], [0.66, st.COLOR_URBANO],
                [1.0, "#16314A"]]


def cargar():
    with open(st.GEOJSON_PROVINCIAS, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    df = pd.read_csv(st.CSV_INDICADORES)
    return geojson, df


def base_layout(titulo, subtitulo):
    return dict(
        title=dict(
            text=f"<b>{titulo}</b><br><sup>{subtitulo}</sup>",
            font=dict(family=st.FONT_FAMILY, size=18, color=st.COLOR_TEXTO_FUERTE),
            x=0.02, xanchor="left", pad=dict(t=10, b=15),
        ),
        font=dict(family=st.FONT_FAMILY, color=st.COLOR_TEXTO),
        margin=dict(l=20, r=20, t=110, b=60),
        paper_bgcolor="white",
        annotations=[dict(
            x=0.02, y=-0.05, xref="paper", yref="paper",
            text=f"<i>{st.FUENTE_NOTA}</i>",
            showarrow=False, font=dict(size=10, color="#7F8C8D"),
            xanchor="left",
        )],
    )


def mapa_choropleth(geojson, df, valor_col, hover_txt, escala_color, titulo,
                    subtitulo, etiqueta_barra, archivo, zmid=None):
    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=df["cod_provincia"],
        z=df[valor_col],
        featureidkey="properties.cod_provincia",
        colorscale=escala_color,
        zmid=zmid,
        marker=dict(line=dict(color="white", width=1.5)),
        colorbar=dict(title=dict(text=etiqueta_barra, side="right"),
                      thickness=14, len=0.62, tickfont=dict(size=11)),
        customdata=df[hover_txt["cols"]].values,
        hovertemplate=hover_txt["template"],
    ))
    fig.update_geos(fitbounds="locations", visible=False,
                    projection_type="transverse mercator")
    fig.update_layout(**base_layout(titulo, subtitulo),
                      width=900, height=820)
    ruta = st.DIR_FINAL / archivo
    fig.write_html(ruta, include_plotlyjs="cdn")
    print(f"  Guardado: {ruta.name}")


def main():
    print("FASE 4 | Generando mapas interactivos (Plotly)")
    geojson, df = cargar()

    mapa_choropleth(
        geojson, df, "pobreza_multi_pct",
        hover_txt=dict(
            cols=["provincia", "region", "pobreza_multi_pct", "poblacion"],
            template=("<b>%{customdata[0]}</b> (Region %{customdata[1]})<br>"
                      "Pobreza multidimensional: %{customdata[2]:.1f}%<br>"
                      "Poblacion: %{customdata[3]:,.0f}<extra></extra>"),
        ),
        escala_color=ESCALA_SECUENCIAL,
        titulo="Pobreza multidimensional en Biobio y Nuble",
        subtitulo="% poblacion en hogares con pobreza multidimensional - CASEN 2024",
        etiqueta_barra="% pobreza multi.",
        archivo="mapa_1_pobreza_multi_interactivo.html",
    )

    media_zona = df["dep_subsidios_pct"].iloc[0] - df["dep_subsidios_brecha"].iloc[0]
    mapa_choropleth(
        geojson, df, "dep_subsidios_brecha",
        hover_txt=dict(
            cols=["provincia", "dep_subsidios_pct", "dep_subsidios_brecha"],
            template=("<b>%{customdata[0]}</b><br>"
                      "Dependencia subsidios: %{customdata[1]:.1f}%<br>"
                      "Brecha vs. macrozona: %{customdata[2]:+.1f} pp"
                      "<extra></extra>"),
        ),
        escala_color=ESCALA_DIVERGENTE, zmid=0,
        titulo="Dependencia de subsidios estatales: brecha territorial",
        subtitulo=(f"Desviacion de cada provincia respecto al promedio de la "
                   f"macrozona ({media_zona:.1f}%) - CASEN 2024"),
        etiqueta_barra="Brecha (pp)",
        archivo="mapa_2_dependencia_subsidios_interactivo.html",
    )

    mapa_choropleth(
        geojson, df, "salud_carencia_pct",
        hover_txt=dict(
            cols=["provincia", "salud_carencia_pct", "salud_carencia_pob"],
            template=("<b>%{customdata[0]}</b><br>"
                      "Carencia acceso a salud: %{customdata[1]:.1f}%<br>"
                      "Personas afectadas: %{customdata[2]:,.0f}"
                      "<extra></extra>"),
        ),
        escala_color=ESCALA_SALUD,
        titulo="Carencia de acceso a salud por provincia",
        subtitulo="% poblacion en hogares con carencia de acceso a salud - CASEN 2024",
        etiqueta_barra="% carencia salud",
        archivo="mapa_4_salud_interactivo.html",
    )

    print("FASE 4 completada.\n")


if __name__ == "__main__":
    main()
