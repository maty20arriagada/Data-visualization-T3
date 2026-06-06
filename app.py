# =============================================================================
#  DASHBOARD INTERACTIVO - Tarea 3 Data Visualization
#  Este archivo contiene la aplicación en Streamlit que renderiza de manera
#  interactiva todos los gráficos para el Deliverable 3.
#  Mantiene fondo blanco, diseño académico, nombres consistentes y sin textos
#  explicativos visibles.
#
#  Revisión 2026-06: aumento de tamaños tipográficos en todos los gráficos
#  (etiquetas dentro de barras, ticks de ejes, leyendas) para mejorar la
#  legibilidad del dashboard sin alterar las proporciones A4 originales.
#  Se incorpora el logo institucional del Dpto. de Ing. Industrial UdeC.
# =============================================================================
from pathlib import Path
import json
import base64

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import streamlit as st

# --- Configuración de la Página ----------------------------------------------
st.set_page_config(
    page_title="Distribución territorial de la pobreza multidimensional en Chile",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Inyección de CSS para fondo blanco y estilo limpio ----------------------
st.markdown(
    """
    <style>
    /* Fondo blanco para todo el contenedor */
    .stApp { background-color: white !important; }

    /* Contenedor principal con padding profesional */
    div.block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.4rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 96%;
    }

    /* Tipografía consistente para encabezados */
    h1, h2, h3, h4 {
        font-family: 'Inter', 'Roboto', sans-serif !important;
        color: #1A2530 !important;
        font-weight: 700 !important;
        letter-spacing: -0.005em;
    }

    /* Subtítulos de sección (markdown h3) un poco más sobrios */
    h3 {
        font-size: 1.05rem !important;
        margin-top: 1.4rem !important;
        margin-bottom: 0.3rem !important;
        padding-bottom: 0.3rem !important;
        border-bottom: 1px solid #ECF0F1;
    }

    /* Espaciado vertical entre filas de gráficos */
    [data-testid="stHorizontalBlock"] {
        margin-bottom: 0.4rem;
    }

    /* Texto del cuerpo con mejor contraste */
    p, label, .stMarkdown { color: #2C3E50; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Constantes y Estilo de la Entrega Anterior ------------------------------
A4_W, A4_H = 620, 506
FONT_FAMILY = "Inter, Roboto, sans-serif"
COLOR_TEXTO_FUERTE = "#1A2530"
COLOR_TEXTO = "#2C3E50"
COLOR_URBANO = "#175884"     # azul (chilenos)
COLOR_RURAL = "#D3541F"      # naranja (inmigrantes)

# --- Tamaños tipográficos centralizados (revisión 2026-06) -------------------
# Se incrementan respecto a la entrega A4 para mejorar la legibilidad del
# dashboard. La proporción del lienzo (620x506) se mantiene.
SZ_TITLE   = 17     # antes 14
SZ_SUBT    = 12     # antes 9.5
SZ_AX_TITLE = 13    # antes 9-10
SZ_AX_TICK  = 12    # antes 8.5-9
SZ_LEGEND   = 12    # antes 8-9
SZ_BAR_TXT  = 14    # antes 9.5-10 (números dentro de barras)
SZ_ANNOT    = 11    # antes 7.5-8 (notas, n=…)
SZ_NOTE     = 10    # antes 7-7.5 (pie de página fino)

PALETA_POBREZA = {
    "Fuera de pobreza":              "#BDC3C7",  # Gris
    "Pobreza por ingresos":          "#F39D2B",  # Amarillo/Naranja
    "Pobreza multidimensional":      "#E64F42",  # Rojo/Coral
    "Pobreza ingresos y multidim.":  "#731819",  # Borgoña/Burgundy
}
ORDEN_POBREZA = list(PALETA_POBREZA.keys())
ORDEN_POBRES = [k for k in ORDEN_POBREZA if k != "Fuera de pobreza"]

PALETA_ZONAS = {
    "Norte Grande":  "#D35400",
    "Norte Chico":   "#F39C12",
    "Zona Central":  "#2B5B84",
    "Zona Sur":      "#117A65",
    "Zona Austral":  "#6C3483",
}
ORDEN_ZONAS = list(PALETA_ZONAS.keys())

# --- Rutas de datos ----------------------------------------------------------
DIR_APP = Path(__file__).resolve().parent
DIR_PROCESSED = DIR_APP / "processed_data"
GEOJSON_PATH = DIR_APP / "Regional.geojson"

# Logo institucional del Dpto. de Ing. Industrial UdeC. Se intenta cargar
# desde varios paths en orden de prioridad: primero los PNG locales del
# proyecto, luego la carpeta personal del usuario (Pictures > Logo
# departamento industrial), y al final el SVG como fallback.
LOGO_CANDIDATES = [
    DIR_APP / "Dpto Ing Industrial (4).png",
    DIR_APP / "Dpto Ing Industrial (1).png",
    DIR_APP / "depto_industrial.png",
    Path(r"C:\Users\Matias Arriagada R\Pictures\Logo departamento industrial\Dpto Ing Industrial (4).png"),
    Path(r"C:\Users\Matias Arriagada R\Pictures\Logo departamento industrial\Dpto Ing Industrial (1).png"),
    Path(r"C:\Users\Matias Arriagada R\Pictures\Logo departamento industrial\Firma DII.png"),
    Path(r"C:\Users\Matias Arriagada R\Pictures\Logo departamento industrial\depto_industrial.png"),
    DIR_APP / "Dpto-Ing-Industrial-_4_.svg",
]


@st.cache_data
def cargar_logo_b64() -> tuple:
    """Devuelve una tupla (mime, base64) del primer logo disponible
    entre los candidatos. Compatible con PNG y SVG. Devuelve
    ("", "") si no encuentra ninguno."""
    for path in LOGO_CANDIDATES:
        if path.exists() and path.is_file():
            data = path.read_bytes()
            ext = path.suffix.lower()
            mime = "image/svg+xml" if ext == ".svg" else "image/png"
            return mime, base64.b64encode(data).decode("ascii")
    return "", ""

# --- Funciones de Formato de la Entrega Anterior -----------------------------
def _titulo_a4(titulo: str, subtitulo: str, size_t: int = None,
                size_s: int = None):
    # Si no se especifica, se usan los tamaños globales (más grandes que
    # los originales A4).
    size_t = SZ_TITLE if size_t is None else size_t
    size_s = SZ_SUBT if size_s is None else size_s
    return dict(
        text=f"<b>{titulo}</b><br><sup>{subtitulo}</sup>",
        font=dict(family=FONT_FAMILY, size=size_t, color=COLOR_TEXTO_FUERTE),
        x=0.02, xanchor="left", y=0.97, yanchor="top",
        pad=dict(t=5, b=5),
    )

def _formato_n(n_muestral: int, n_ponderado: float, unidad: str = "hogares") -> str:
    return (f"n = {n_ponderado:,.0f} {unidad} (población expandida) · "
            f"{n_muestral:,} {unidad} en la muestra")

def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{alpha})"

# --- Carga de datos auxiliares -----------------------------------------------
@st.cache_data
def cargar_geojson():
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf["codregion"] = gdf["codregion"].astype(int)
    # Excluir la zona sin demarcar (código 0)
    gdf = gdf[gdf["codregion"] != 0].copy()
    
    # Limpiar geometría de la Región de Valparaíso (código 5) para excluir islas
    idx_reg5 = gdf[gdf["codregion"] == 5].index
    if not idx_reg5.empty:
        from shapely.geometry import MultiPolygon
        geom = gdf.loc[idx_reg5[0], "geometry"]
        if isinstance(geom, MultiPolygon):
            continental_polys = [p for p in geom.geoms if p.bounds[0] > -73.0]
            gdf.loc[idx_reg5[0], "geometry"] = MultiPolygon(continental_polys)
            
    return gdf

def cargar_datos_procesados():
    g1 = pd.read_csv(DIR_PROCESSED / "g1_donut.csv")
    g2_macro = pd.read_csv(DIR_PROCESSED / "g2_macrozonas.csv")
    regional = pd.read_csv(DIR_PROCESSED / "regional_poverty.csv")
    g3 = pd.read_csv(DIR_PROCESSED / "g3_sankey.csv")
    g4 = pd.read_csv(DIR_PROCESSED / "g4_radar.csv")
    g5 = pd.read_csv(DIR_PROCESSED / "g5_historical_inmig.csv")
    moran_lisa = pd.read_csv(DIR_PROCESSED / "moran_lisa.csv")

    with open(DIR_PROCESSED / "moran_stats.json", "r") as f:
        moran_stats = json.load(f)

    return g1, g2_macro, regional, g3, g4, g5, moran_lisa, moran_stats


@st.cache_data
def cargar_lisa_comunal():
    """Carga el GeoJSON con la clasificacion LISA por comuna del Norte
    Grande (pre-procesado por zip/preprocess_H_lisa_comunal.py).
    Devuelve un GeoDataFrame en EPSG:4326 listo para Plotly."""
    path = DIR_PROCESSED / "H_lisa_norte_grande.geojson"
    if not path.exists():
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(path)
    return gdf

# Cargar todos los recursos
gdf_chile = cargar_geojson()
df_g1, df_g2_macro, df_regional, df_g3, df_g4, df_g5, df_spatial, stats_moran = cargar_datos_procesados()
gdf_lisa_ng = cargar_lisa_comunal()

# =============================================================================
# FUNCIONES DE PLOTEO (A4 Fieles al Reporte de la Entrega Anterior)
# =============================================================================

def plot_a4_g1_donut(df_g1):
    df_g1["estado_pob"] = pd.Categorical(
        df_g1["estado_pob"], 
        categories=list(PALETA_POBREZA.keys()), 
        ordered=True
    )
    df_g1 = df_g1.sort_values("estado_pob").reset_index(drop=True)
    df_g1["pct"] = df_g1["expr"] / df_g1["expr"].sum() * 100
    
    hog_pob = df_g1[df_g1["estado_pob"] != "Fuera de pobreza"]["expr"].sum()
    pct_pob = hog_pob / df_g1["expr"].sum() * 100
    
    # n muestral nacional se calcula dinámicamente
    n_mues = int(df_g1["n"].sum()) if "n" in df_g1.columns else 78654
    n_pond = float(df_g1["expr"].sum())
    n_label = _formato_n(n_mues, n_pond)
    
    etq = [f"<b>{r.estado_pob}</b><br>{r.pct:.1f}%" for r in df_g1.itertuples()]
    colors_g1 = [PALETA_POBREZA[k] for k in df_g1["estado_pob"]]
    
    fig = go.Figure(go.Pie(
        labels=df_g1["estado_pob"].tolist(), values=df_g1["expr"].tolist(), hole=0.55,
        sort=False, direction="clockwise",
        marker=dict(colors=colors_g1, line=dict(color="white", width=1.5)),
        text=etq, textinfo="text", textposition="outside",
        textfont=dict(family=FONT_FAMILY, size=SZ_BAR_TXT - 1,
                      color=COLOR_TEXTO_FUERTE),
        hovertemplate=("<b>%{label}</b><br>%{value:,.0f} hogares<br>"
                       "%{percent}<extra></extra>"),
        pull=[0, 0.01, 0.01, 0.02], showlegend=False,
    ))
    fig.add_annotation(x=0.5, y=0.56, xref="paper", yref="paper",
                       text=f"<b>{pct_pob:.1f}%</b>",
                       font=dict(family=FONT_FAMILY, size=36, color="#78281F"),
                       showarrow=False)
    fig.add_annotation(x=0.5, y=0.40, xref="paper", yref="paper",
                       text=("hogares en<br>alguna pobreza"),
                       font=dict(family=FONT_FAMILY, size=SZ_AX_TICK,
                                 color=COLOR_TEXTO),
                       showarrow=False, align="center")
    fig.update_layout(
        title=_titulo_a4(
            "A. Distribución nacional de los hogares según condición de pobreza",
            "% de hogares según cruce pobreza por ingresos × pobreza multidimensional (CASEN 2024)."
        ),
        margin=dict(t=95, b=45, l=70, r=70),
        paper_bgcolor="white", plot_bgcolor="white",
        width=A4_W, height=A4_H,
        font=dict(family=FONT_FAMILY, color=COLOR_TEXTO, size=SZ_AX_TICK),
        annotations=list(fig.layout.annotations) + [
            dict(x=0.5, y=-0.05, xref="paper", yref="paper",
                 text=f"<i>{n_label}</i>", showarrow=False,
                 xanchor="center",
                 font=dict(size=SZ_ANNOT, color="#7F8C8D",
                           family=FONT_FAMILY))],
    )
    return fig


def plot_a4_g2_macrozonas(df_g2_macro):
    # Tabular la composición de manera explícita y robusta
    df_pobres = df_g2_macro[df_g2_macro["estado_pob"] != "Fuera de pobreza"].copy()
    
    tabla_expr = df_pobres.groupby(["zona", "estado_pob"])["expr"].sum().unstack(fill_value=0)
    tabla_expr = tabla_expr.reindex(columns=ORDEN_POBRES, fill_value=0)
    tabla_expr = tabla_expr.reindex(index=ORDEN_ZONAS, fill_value=0)
    
    sumas_zona = tabla_expr.sum(axis=1)
    sumas_zona = sumas_zona.replace(0, 1) # Evitar división por cero
    pct = tabla_expr.div(sumas_zona, axis=0) * 100
    
    # Calcular n y N para el pie (hogares pobres de la muestra y expandidos)
    n_mues = int(df_g2_macro["n"].sum()) if "n" in df_g2_macro.columns else int(df_regional["n_sample_poor"].sum())
    n_pond = float(df_g2_macro["expr"].sum())
    n_label = _formato_n(n_mues, n_pond, "hogares pobres")
    
    fig = go.Figure()
    for estado in ORDEN_POBRES:
        v = list(pct[estado].values)
        fig.add_trace(go.Bar(
            x=v, y=ORDEN_ZONAS, orientation="h", name=estado,
            marker=dict(color=PALETA_POBREZA[estado],
                        line=dict(color="white", width=0.6)),
            text=[f"{x:.0f}%" if x >= 9 else "" for x in v],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(family=FONT_FAMILY, size=SZ_BAR_TXT,
                          color="white"),
            hovertemplate=(f"<b>{estado}</b><br>%{{y}}<br>"
                           "%{x:.2f}%<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack",
        title=_titulo_a4(
            "B. Composición de la pobreza por macrozona",
            "% de hogares pobres por tipo, normalizado al 100% intra-zona (excluye fuera de pobreza) (CASEN 2024)."
        ),
        xaxis=dict(title="% hogares pobres",
                   ticksuffix="%", range=[0, 100], gridcolor="#ECF0F1",
                   tickfont=dict(size=SZ_AX_TICK),
                   title_font=dict(size=SZ_AX_TITLE)),
        yaxis=dict(title="", autorange="reversed",
                   tickfont=dict(size=SZ_AX_TICK,
                                 color=COLOR_TEXTO_FUERTE)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5,
                    font=dict(size=SZ_LEGEND, family=FONT_FAMILY),
                    traceorder="reversed"),
        font=dict(family=FONT_FAMILY, color=COLOR_TEXTO),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=70, b=125, l=110, r=25),
        width=A4_W, height=A4_H,
        annotations=[dict(
            x=0.5, y=-0.34, xref="paper", yref="paper",
            text=f"<i>{n_label}</i>", showarrow=False,
            xanchor="center",
            font=dict(size=SZ_ANNOT, color="#7F8C8D",
                      family=FONT_FAMILY))],
    )
    return fig


def plot_a4_g10_sankey(df_ng):
    nodos_pobreza = ORDEN_POBRES
    nodos_origen = ["Hogares chilenos", "Hogares inmigrantes"]
    nodos_trifecta = ["Escolaridad", "Déficit cuantitativo", "Conectividad digital", "Informalidad", "Otras carencias", "Sin carencias"]
    nodos_ids = nodos_pobreza + nodos_origen + nodos_trifecta
    idx = {n: i for i, n in enumerate(nodos_ids)}
    
    ETIQ_A4 = {
        "Pobreza por ingresos":            "Pobreza<br>por ingresos",
        "Pobreza multidimensional":        "Pobreza<br>multidimensional",
        "Pobreza ingresos y multidim.":    "Ingresos +<br>multidim.",
        "Hogares chilenos":                "Hogares<br>chilenos",
        "Hogares inmigrantes":             "Hogares<br>inmigrantes",
        "Escolaridad":                     "Escolaridad<br><sup>(adulto con menos años que los exigidos)</sup>",
        "Déficit cuantitativo":            "Déficit cuantitativo<br><sup>(irrecuperable, allegamiento o hacinamiento)</sup>",
        "Conectividad digital":            "Conectividad digital<br><sup>(sin internet fijo ni dispositivo)</sup>",
        "Informalidad":                    "Informalidad<br><sup>(trabajador sin cotización previsional)</sup>",
        "Otras carencias":                 "Otras carencias<br><sup>(alguna de las 16 restantes)</sup>",
        "Sin carencias":                   "Sin carencias<br><sup>(pobre solo por ingresos)</sup>",
    }
    nodos_labels = [ETIQ_A4.get(n, n) for n in nodos_ids]

    f1 = df_ng.groupby(["estado_pob", "origen_jefe"])["expr"].sum().reset_index().dropna()
    f1 = f1[f1["estado_pob"].isin(nodos_pobreza)]
    src1 = f1["estado_pob"].map(idx).tolist()
    tgt1 = f1["origen_jefe"].map(idx).tolist()
    val1 = f1["expr"].tolist()
    col1 = [hex_to_rgba(PALETA_POBREZA[s], 0.42) for s in f1["estado_pob"]]

    f2 = df_ng.groupby(["origen_jefe", "trifecta"])["expr"].sum().reset_index().dropna()
    src2 = f2["origen_jefe"].map(idx).tolist()
    tgt2 = f2["trifecta"].map(idx).tolist()
    val2 = f2["expr"].tolist()
    col_orig = {
        "Hogares chilenos":    "rgba(189, 195, 199, 0.20)",
        "Hogares inmigrantes": "rgba(211, 84, 0, 0.75)",
    }
    col2 = [col_orig[o] for o in f2["origen_jefe"]]

    CARENCIAS_OUTLIER = ["Escolaridad", "Déficit cuantitativo", "Conectividad digital", "Informalidad"]
    tot_inm = df_ng.loc[df_ng["origen_jefe"] == "Hogares inmigrantes", "expr"].sum()
    inm_out = df_ng.loc[(df_ng["origen_jefe"] == "Hogares inmigrantes") & (df_ng["trifecta"].isin(CARENCIAS_OUTLIER)), "expr"].sum()
    tot_chi = df_ng.loc[df_ng["origen_jefe"] == "Hogares chilenos", "expr"].sum()
    chi_out = df_ng.loc[(df_ng["origen_jefe"] == "Hogares chilenos") & (df_ng["trifecta"].isin(CARENCIAS_OUTLIER)), "expr"].sum()
    
    pct_inm_out = 100 * inm_out / tot_inm
    pct_chi_out = 100 * chi_out / tot_chi

    n_mues = 2613
    n_pond = float(df_ng["expr"].sum())
    n_label = _formato_n(n_mues, n_pond, "hogares pobres")

    color_nodos = (
        [PALETA_POBREZA[k] for k in nodos_pobreza]
        + [COLOR_URBANO, COLOR_RURAL]
        + ["#7F8C8D", "#7F8C8D", "#7F8C8D", "#7F8C8D", "#95A5A6", "#D5DBDB"]
    )

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodos_labels, color=color_nodos,
                  pad=14, thickness=16,
                  line=dict(color="white", width=0.4),
                  hovertemplate=("<b>%{label}</b><br>"
                                 "Hogares: %{value:,.0f}<extra></extra>")),
        link=dict(source=src1 + src2, target=tgt1 + tgt2,
                  value=val1 + val2, color=col1 + col2,
                  hovertemplate=("%{source.label} → %{target.label}<br>"
                                 "Hogares: %{value:,.0f}<extra></extra>")),
    ))
    subtitulo = (f"<b>{pct_inm_out:.0f}%</b> de inmigrantes pobres vs <b>{pct_chi_out:.0f}%</b> de chilenos pobres en ≥1 "
                 "de las 4 carencias críticas (metodología 2024).")

    fig.update_layout(
        title=_titulo_a4(
            "D. Flujos de pobreza y carencias en hogares de Norte Grande",
            subtitulo, size_t=SZ_TITLE, size_s=SZ_SUBT),
        # Tamaño base de las etiquetas de nodos del Sankey
        font=dict(family=FONT_FAMILY, color=COLOR_TEXTO_FUERTE,
                  size=SZ_AX_TICK - 1),
        margin=dict(t=70, b=48, l=12, r=12),
        paper_bgcolor="white", plot_bgcolor="white",
        width=A4_W, height=A4_H,
        annotations=[dict(
            x=0.5, y=-0.05, xref="paper", yref="paper",
            text=f"<i>{n_label}</i>", showarrow=False,
            xanchor="center",
            font=dict(size=SZ_ANNOT, color="#7F8C8D",
                      family=FONT_FAMILY))],
    )
    return fig

def plot_a4_g3_barras(df_ng):
    """E. Composición de la pobreza según origen del hogar en Norte Grande.
    Barras apiladas al 100% intra-grupo: hogares chilenos vs inmigrantes
    cruzados con los 3 tipos de pobreza (excluye fuera de pobreza).
    """
    # --- Agrupar y sumar ponderadores -----------------------------------
    tab = (df_ng.groupby(["origen_jefe", "estado_pob"])["expr"].sum()
           .unstack(fill_value=0)
           .reindex(columns=ORDEN_POBRES, fill_value=0)
           .reindex(index=["Hogares chilenos", "Hogares inmigrantes"],
                    fill_value=0))

    # --- Normalización intra-grupo a porcentajes (0–100) ---------------
    suma_por_fila = tab.sum(axis=1).replace(0, 1)
    pct = (tab.div(suma_por_fila, axis=0) * 100).fillna(0)

    # Categorías del eje X en orden fijo (lista pura, evita problemas de
    # serialización de pd.Index entre múltiples traces apilados).
    categorias_x = ["Hogares chilenos", "Hogares inmigrantes"]

    n_mues = 2613
    n_pond = float(df_ng["expr"].sum())
    n_label = _formato_n(n_mues, n_pond)

    TEXTO_OSCURO = "#1A1A1A"
    TEXTO_GRIS = "#4A4A4A"

    fig = go.Figure()
    for estado in ORDEN_POBRES:
        valores = [float(pct.loc[cat, estado]) for cat in categorias_x]
        textos = [f"<b>{v:.1f}%</b>" if v >= 3 else "" for v in valores]
        fig.add_trace(go.Bar(
            x=categorias_x, y=valores, name=estado,
            marker=dict(color=PALETA_POBREZA[estado],
                        line=dict(color="white", width=0.8)),
            text=textos,
            textposition="inside", insidetextanchor="middle",
            textfont=dict(family=FONT_FAMILY, size=SZ_BAR_TXT,
                          color="white"),
            hovertemplate=(f"<b>{estado}</b><br>"
                           "Origen: %{x}<br>"
                           "Participación: %{y:.2f}%<extra></extra>"),
        ))

    fig.update_layout(
        barmode="stack",
        bargap=0.45,
        title=_titulo_a4(
            "E. Composición de la pobreza según origen del hogar en Norte Grande",
            "% de hogares pobres normalizado al 100% intra-grupo (chilenos vs inmigrantes), CASEN 2024."
        ),
        xaxis=dict(
            title="",
            type="category",
            categoryorder="array",
            categoryarray=categorias_x,
            tickfont=dict(size=SZ_AX_TICK + 2, color=TEXTO_OSCURO,
                          family=FONT_FAMILY),
        ),
        yaxis=dict(
            title="% de hogares pobres",
            ticksuffix="%",
            range=[0, 100],
            gridcolor="#ECF0F1",
            tickfont=dict(size=SZ_AX_TICK, color=TEXTO_OSCURO,
                          family=FONT_FAMILY),
            title_font=dict(size=SZ_AX_TITLE, color=TEXTO_OSCURO,
                            family=FONT_FAMILY),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20,
            xanchor="center", x=0.5,
            font=dict(size=SZ_LEGEND, family=FONT_FAMILY,
                      color=TEXTO_OSCURO),
            traceorder="reversed",
        ),
        font=dict(family=FONT_FAMILY, color=TEXTO_OSCURO),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=80, b=110, l=75, r=35),
        width=A4_W, height=A4_H,
        annotations=[
            dict(x=0.5, y=-0.34, xref="paper", yref="paper",
                 text=f"<i>{n_label}</i>", showarrow=False,
                 xanchor="center",
                 font=dict(size=SZ_ANNOT, color=TEXTO_GRIS,
                           family=FONT_FAMILY)),
        ],
    )
    return fig

def plot_a4_g4_radar(df_radar):
    dims = ["Educacion", "Salud", "Trabajo y Seg. Social", "Vivienda y Entorno", "Redes y Cohesion"]
    ejes = dims + [dims[0]]
    
    resultados = {}
    for _, row in df_radar.iterrows():
        g = row["origen_jefe"]
        resultados[g] = {
            "scores": {d: row[d] for d in dims},
            "n_hog": int(row["n_hog"]),
            "n_pond": float(row["n_pond"])
        }
        
    chil = resultados["Hogares chilenos"]["scores"]
    inmi = resultados["Hogares inmigrantes"]["scores"]
    chil_v = [chil[d] for d in dims] + [chil[dims[0]]]
    inmi_v = [inmi[d] for d in dims] + [inmi[dims[0]]]
    
    max_obs = max(max(chil.values()), max(inmi.values()))
    escala_max = max(0.5, np.ceil(max_obs * 1.25 * 2) / 2)
    paso = 0.5 if escala_max <= 2 else 1

    tickvals = [0.5, 1.0]
    ticktext = ["0,5 carencias", "1,0 carencia"]

    n_mues = 9373
    n_pond = sum(r["n_pond"] for r in resultados.values())
    n_label = _formato_n(n_mues, n_pond)

    # --- COLORES DE TEXTO DE ALTO CONTRASTE ---
    TEXTO_OSCURO = "#1A1A1A"   # casi negro, máximo contraste sobre blanco
    TEXTO_GRIS = "#4A4A4A"     # para notas secundarias, pero legible

    fig = go.Figure()
    # Traza Hogares Chilenos (azul)
    fig.add_trace(go.Scatterpolar(
        r=chil_v, theta=ejes, mode="lines+markers",
        name=f"Hogares Chilenos (jefe nacido en Chile) (n={resultados['Hogares chilenos']['n_hog']:,})",
        line=dict(color=COLOR_URBANO, width=2.4),
        marker=dict(size=6, color=COLOR_URBANO, symbol="circle",
                    line=dict(color="white", width=0.8)),
        fill="toself", fillcolor="rgba(43,91,132,0.22)",
        hovertemplate="<b>%{theta}</b><br>Hogares Chilenos: %{r:.3f}<extra></extra>",
    ))
    # Traza Hogares Inmigrantes (naranja)
    fig.add_trace(go.Scatterpolar(
        r=inmi_v, theta=ejes, mode="lines+markers",
        name=f"Hogares Inmigrantes (jefe nacido en el extranjero) (n={resultados['Hogares inmigrantes']['n_hog']:,})",
        line=dict(color=COLOR_RURAL, width=2.4),
        marker=dict(size=6, color=COLOR_RURAL, symbol="square",
                    line=dict(color="white", width=0.8)),
        fill="toself", fillcolor="rgba(211,84,0,0.32)",
        hovertemplate="<b>%{theta}</b><br>Hogares Inmigrantes: %{r:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor="#FAFAFA",
            radialaxis=dict(
                visible=True, range=[0, escala_max],
                tickvals=list(tickvals),
                ticktext=ticktext,
                tickfont=dict(size=SZ_AX_TICK - 1, color=TEXTO_OSCURO,
                              family=FONT_FAMILY),
                angle=30, tickangle=0,
                gridcolor="#E5E7E9", linecolor="#E5E7E9"
            ),
            angularaxis=dict(
                tickfont=dict(size=SZ_AX_TICK + 1, color=TEXTO_OSCURO,
                              family=FONT_FAMILY),
                rotation=90, direction="clockwise",
                linecolor="#BDC3C7", gridcolor="#E5E7E9"
            ),
        ),
        title=_titulo_a4(
            "F. Intensidad de carencias por dimensión en Norte Grande",
            "Promedio ponderado del número de carencias de la metodología multidimensional 2015."
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.14,
            xanchor="center", x=0.5,
            font=dict(size=SZ_LEGEND, family=FONT_FAMILY,
                      color=TEXTO_OSCURO),
            title=dict(text="Tipos de hogar", side="top",
                       font=dict(size=SZ_LEGEND - 1,
                                 color=TEXTO_OSCURO))
        ),
        font=dict(family=FONT_FAMILY, color=TEXTO_OSCURO),
        margin=dict(t=65, b=135, l=55, r=55),
        height=A4_H, width=A4_W,
        paper_bgcolor="white", plot_bgcolor="white",
        annotations=[
            dict(x=0.5, y=-0.22, xref="paper", yref="paper",
                 text="Unidad: carencias promedio por hogar.",
                 showarrow=False, xanchor="center",
                 font=dict(size=SZ_ANNOT, color=TEXTO_GRIS,
                           family=FONT_FAMILY)),
            dict(x=0.5, y=-0.32, xref="paper", yref="paper",
                 text=f"<i>{n_label}</i>", showarrow=False,
                 xanchor="center",
                 font=dict(size=SZ_ANNOT, color=TEXTO_GRIS,
                           family=FONT_FAMILY))
        ],
    )
    return fig

def plot_a4_g5_lineas(df_serie):
    n_pond_2024 = 7143171
    n_mues_2024 = 78654
    n_label = (f"n CASEN 2024 = {n_pond_2024:,.0f} hogares (población "
               f"expandida) · {n_mues_2024:,} hogares en la muestra")

    fig = go.Figure()
    for zona in ORDEN_ZONAS:
        sub = df_serie[df_serie["zona"] == zona].sort_values("anio")
        fig.add_trace(go.Scatter(
            x=sub["anio"].tolist(), y=sub["pct_inmig"].tolist(),
            mode="lines+markers", name=zona,
            line=dict(color=PALETA_ZONAS[zona], width=2.5),
            marker=dict(size=8, color=PALETA_ZONAS[zona],
                        line=dict(color="white", width=0.8)),
            hovertemplate=(f"<b>{zona}</b><br>CASEN %{{x}}<br>"
                           "%{y:.2f}%<extra></extra>"),
        ))

    fin = df_serie[df_serie["anio"] == 2024].sort_values("pct_inmig",
                                                          ascending=False)
    for _, r in fin.iterrows():
        fig.add_annotation(
            x=r["anio"], y=r["pct_inmig"],
            text=f"<b>{r['pct_inmig']:.1f}%</b>",
            showarrow=False, xanchor="left", xshift=5,
            font=dict(family=FONT_FAMILY, size=SZ_AX_TICK,
                      color=PALETA_ZONAS[r["zona"]]))
    y_max = df_serie["pct_inmig"].max() * 1.30

    fig.update_layout(
        title=_titulo_a4(
            "G. Evolución de hogares inmigrantes por macrozona",
            "% de hogares con jefe nacido en el extranjero (CASEN 2013-2024)."
        ),
        xaxis=dict(title="Año",
                   tickmode="array",
                   tickvals=[2013, 2015, 2017, 2022, 2024],
                   ticktext=["2013", "2015", "2017", "2022", "2024"],
                   range=[2012, 2026.5], gridcolor="#ECF0F1",
                   tickfont=dict(size=SZ_AX_TICK),
                   title_font=dict(size=SZ_AX_TITLE)),
        yaxis=dict(title="% hogares inmigrantes",
                   ticksuffix="%", range=[0, y_max],
                   gridcolor="#ECF0F1",
                   tickfont=dict(size=SZ_AX_TICK),
                   title_font=dict(size=SZ_AX_TITLE)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.20,
                    xanchor="center", x=0.5,
                    font=dict(size=SZ_LEGEND, family=FONT_FAMILY)),
        font=dict(family=FONT_FAMILY, color=COLOR_TEXTO),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=70, b=125, l=70, r=35),
        width=A4_W, height=A4_H,
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.34, xref="paper", yref="paper",
            text=f"<i>{n_label}</i>", showarrow=False,
            xanchor="center",
            font=dict(size=SZ_ANNOT, color="#7F8C8D",
                      family=FONT_FAMILY))],
    )
    return fig


# =============================================================================
# REEMPLAZO DE GRÁFICO B: 5 MAPAS GEOGRÁFICOS POR ZONA
# =============================================================================

escalas = {
    "Pobreza por ingresos": [[0, "#F39D2B"], [1, "#F39D2B"]],
    "Pobreza multidimensional": [[0, "#E64F42"], [1, "#E64F42"]],
    "Pobreza ingresos y multidim.": [[0, "#731819"], [1, "#731819"]]
}

def plot_zone_map(zona_name, gdf_chile, df_regional):
    df_z = df_regional[df_regional["zona"] == zona_name].copy()
    gdf_z = gdf_chile[gdf_chile["codregion"].isin(df_z["region"])].copy()
    gdf_z = gdf_z.merge(df_z, left_on="codregion", right_on="region")
    
    fig = go.Figure()
    
    for pob_tipo, scale in escalas.items():
        sub_gdf = gdf_z[gdf_z["cat_sobrerep"] == pob_tipo]
        if len(sub_gdf) == 0:
            continue
            
        custom_data_sub = sub_gdf[[
            "region_name", 
            "cat_sobrerep", 
            "diff_ingresos", 
            "diff_multi", 
            "diff_ambas",
            "n_sample_tot",
            "n_expanded_tot",
            "n_sample_poor",
            "n_expanded_poor",
            "val_sobrerep"
        ]].values
        
        fig.add_trace(go.Choropleth(
            geojson=json.loads(sub_gdf.to_json()),
            locations=sub_gdf["region"].tolist(),
            featureidkey="properties.codregion",
            z=[1] * len(sub_gdf),
            colorscale=scale,
            marker=dict(line=dict(color="white", width=1.5)),
            showscale=False,
            customdata=custom_data_sub,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Máx Sobrerrepresentación: <b>%{customdata[1]}</b> (+%{customdata[9]:.2f}%)<br>"
                "Diferencias vs Promedio Nacional:<br>"
                " • Pobreza por ingresos: %{customdata[2]:+.2f}%<br>"
                " • Pobreza multidimensional: %{customdata[3]:+.2f}%<br>"
                " • Pobreza ingresos y multidim. (ambas): %{customdata[4]:+.2f}%<br>"
                "Tamaño de muestra:<br>"
                " • Muestra total (n): %{customdata[5]:,.0f} hog. (%{customdata[6]:,.0f} expandidos)<br>"
                " • Muestra pobres (n_pob): %{customdata[7]:,.0f} hog. (%{customdata[8]:,.0f} expandidos)<br>"
                "<extra></extra>"
            )
        ))
        
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="transverse mercator"
    )
    
    fig.update_layout(
        title=dict(
            text=f"<b>{zona_name}</b>",
            font=dict(family=FONT_FAMILY, size=SZ_AX_TITLE + 2,
                      color=COLOR_TEXTO_FUERTE),
            x=0.02, xanchor="left", y=0.96, yanchor="top"
        ),
        margin=dict(l=5, r=5, t=40, b=5),
        paper_bgcolor="white",
        width=200,
        height=320
    )
    return fig


# =============================================================================
# NUEVOS GRÁFICOS: Moran Scatterplot y LISA Cluster Map
# =============================================================================

def plot_moran_scatterplot(df_spatial, stats_moran):
    col_quads = {
        "HH": "#E74C3C",  # Rojo
        "LH": "#85C1E9",  # Azul claro
        "LL": "#2E86C1",  # Azul oscuro
        "HL": "#F5B7B1"   # Coral/Rosa
    }

    x_line = np.array([df_spatial["y_std"].min() - 0.2, df_spatial["y_std"].max() + 0.2])
    y_line = stats_moran["slope"] * x_line + stats_moran["intercept"]

    fig = go.Figure()

    # Trendline
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=y_line.tolist(),
        mode="lines",
        name="Línea de tendencia",
        line=dict(color="#7F8C8D", width=1.5, dash="dash"),
        hoverinfo="skip"
    ))

    # Points by quadrant
    for q, color in col_quads.items():
        sub = df_spatial[df_spatial["quadrant"] == q]
        if len(sub) == 0:
            continue
        
        fig.add_trace(go.Scatter(
            x=sub["y_std"].tolist(), y=sub["spatial_lag"].tolist(),
            mode="markers",
            name=q,
            marker=dict(size=9, color=color, line=dict(color="white", width=0.8)),
            customdata=sub[["region_name", "pobreza_multi_pct"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pobreza multidimensional: %{customdata[1]:.2f}%<br>"
                "Pobreza estandarizada: %{x:.2f}<br>"
                "Rezago espacial: %{y:.2f}<extra></extra>"
            ),
            showlegend=False
        ))

    fig.add_vline(x=0, line=dict(color="#BDC3C7", width=1))
    fig.add_hline(y=0, line=dict(color="#BDC3C7", width=1))

    fig.add_annotation(x=2.0, y=1.8, text="<b>HH</b> (Alto-Alto)",
                       showarrow=False,
                       font=dict(color="#E74C3C", size=SZ_AX_TICK + 1,
                                 family=FONT_FAMILY))
    fig.add_annotation(x=-1.2, y=1.8, text="<b>LH</b> (Bajo-Alto)",
                       showarrow=False,
                       font=dict(color="#85C1E9", size=SZ_AX_TICK + 1,
                                 family=FONT_FAMILY))
    fig.add_annotation(x=-1.2, y=-1.3, text="<b>LL</b> (Bajo-Bajo)",
                       showarrow=False,
                       font=dict(color="#2E86C1", size=SZ_AX_TICK + 1,
                                 family=FONT_FAMILY))
    fig.add_annotation(x=2.0, y=-1.3, text="<b>HL</b> (Alto-Bajo)",
                       showarrow=False,
                       font=dict(color="#F5B7B1", size=SZ_AX_TICK + 1,
                                 family=FONT_FAMILY))

    fig.add_annotation(
        x=0.05, y=0.92, xref="paper", yref="paper",
        text=f"Moran's I: <b>{stats_moran['moran_i']:.3f}</b><br>p-value: <b>{stats_moran['p_value']:.4f}</b>",
        showarrow=False,
        align="left",
        bgcolor="white",
        bordercolor="#BDC3C7",
        borderwidth=1,
        borderpad=8,
        font=dict(size=SZ_AX_TITLE, family=FONT_FAMILY)
    )

    fig.update_layout(
        title=_titulo_a4(
            "H. Autocorrelación espacial global de la pobreza multidimensional",
            "Gráfico de dispersión de Moran: Pobreza regional estandarizada vs Rezago espacial (CASEN 2024)."
        ),
        xaxis=dict(title="Pobreza regional estandarizada (Z-score)",
                   gridcolor="#F2F3F4",
                   tickfont=dict(size=SZ_AX_TICK),
                   title_font=dict(size=SZ_AX_TITLE)),
        yaxis=dict(title="Rezago espacial de la pobreza",
                   gridcolor="#F2F3F4",
                   tickfont=dict(size=SZ_AX_TICK),
                   title_font=dict(size=SZ_AX_TITLE)),
        font=dict(family=FONT_FAMILY, color=COLOR_TEXTO),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=75, b=50, l=70, r=45),
        width=A4_W,
        height=A4_H,
    )
    return fig


def plot_lisa_comunal_norte_grande(gdf_lisa_ng):
    """H. Clusteres espaciales de pobreza por ingresos Y multidimensional
    (simultaneamente) en las comunas del Norte Grande.

    El cálculo LISA (Moran Local con vecindad Queen → KNN=4 por la
    presencia de islas, y semilla 42) se realiza previamente en
    ``zip/preprocess_H_lisa_comunal.py``. Esta función solo renderiza
    el GeoJSON pre-procesado con Plotly Choropleth para que el dashboard
    cargue rápidamente y conserve la interactividad (tooltips por comuna)."""
    # Paleta exacta del script original (vino tinto -> rosa palido)
    paleta_lisa = {
        "Alto-Alto":         "#731819",
        "Bajo-Bajo":         "#A64B5A",
        "Alto-Bajo":         "#C07884",
        "Bajo-Alto":         "#E4B7BB",
        "No significativo":  "#D9D9D9",
        "Sin datos":         "#F8F9F9",
    }
    orden_cats = ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto",
                  "No significativo", "Sin datos"]

    if gdf_lisa_ng is None or len(gdf_lisa_ng) == 0:
        # Fallback informativo: el GeoJSON no se encontro
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text=("<b>Mapa LISA no disponible</b><br>"
                  "<i>Ejecuta primero "
                  "<code>zip/preprocess_H_lisa_comunal.py</code> para "
                  "generar <code>processed_data/"
                  "H_lisa_norte_grande.geojson</code>.</i>"),
            showarrow=False, font=dict(size=12, color="#7F8C8D",
                                        family=FONT_FAMILY),
            align="center")
        fig.update_layout(width=A4_W, height=A4_H,
                          paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Asegurar un identificador unico por comuna
    gdf = gdf_lisa_ng.copy()
    gdf["id_comuna"] = gdf.index.astype(str)
    geojson = json.loads(gdf.to_json())

    fig = go.Figure()
    for cat in orden_cats:
        sub = gdf[gdf["lisa_cat"] == cat]
        if len(sub) == 0:
            continue
        color = paleta_lisa[cat]
        customdata = sub[[
            "Comuna", "lisa_cat", "pct_pobreza", "lisa_p_value",
            "n_sample", "n_sample_poor", "pobre_pond", "total_pond",
        ]].fillna(-1).values

        fig.add_trace(go.Choropleth(
            geojson=geojson,
            locations=sub["id_comuna"].tolist(),
            featureidkey="properties.id_comuna",
            z=[1] * len(sub),
            colorscale=[[0, color], [1, color]],
            marker=dict(line=dict(color="white", width=0.8)),
            showscale=False,
            name=cat,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Clúster LISA: <b>%{customdata[1]}</b><br>"
                "Pobreza doble (ingresos × multi): %{customdata[2]:.2f}%<br>"
                "p-value local: %{customdata[3]:.4f}<br>"
                "Muestra: %{customdata[4]:,.0f} hog. "
                "(pobres dobles: %{customdata[5]:,.0f})<br>"
                "Población ponderada: %{customdata[7]:,.0f} hog.<extra></extra>"
            ),
            showlegend=True,
        ))

    fig.update_geos(
        fitbounds="locations", visible=False,
        projection_type="transverse mercator",
        bgcolor="white",
    )
    fig.update_layout(
        title=_titulo_a4(
            "H. Clústeres espaciales de pobreza por ingresos y multidimensional",
            "Mapa LISA comunal del Norte Grande: pobreza simultánea por ingresos y multidimensional (CASEN 2024)."
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.06,
            xanchor="center", x=0.5,
            font=dict(size=SZ_LEGEND - 1, family=FONT_FAMILY,
                      color=COLOR_TEXTO_FUERTE),
            title=dict(text="Clasificación LISA",
                       font=dict(size=SZ_LEGEND - 1,
                                 family=FONT_FAMILY,
                                 color=COLOR_TEXTO_FUERTE)),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=10, r=10, t=78, b=72),
        paper_bgcolor="white", plot_bgcolor="white",
        width=A4_W, height=A4_H,
        annotations=[dict(
            x=0.5, y=-0.20, xref="paper", yref="paper",
            text=("<i>Nota: la desagregación comunal no es representativa "
                  "en CASEN; permite observar el patrón territorial "
                  "general. Vecindad: Queen → KNN=4 (por islas); "
                  "p ≤ 0,05.</i>"),
            showarrow=False, xanchor="center",
            font=dict(size=SZ_NOTE, color="#7F8C8D",
                      family=FONT_FAMILY)),
        ],
    )
    return fig


# =============================================================================
# ESTRUCTURA E INTERFAZ DEL DASHBOARD
# =============================================================================

# --- Header institucional: logo + título + autoría ---------------------------
LOGO_MIME, LOGO_B64 = cargar_logo_b64()
if LOGO_B64:
    logo_img_html = (f'<img src="data:{LOGO_MIME};base64,{LOGO_B64}" '
                     'style="height: 92px; width: auto; max-width: 280px; '
                     'object-fit: contain; '
                     'filter: drop-shadow(0 1px 0 rgba(0,0,0,0.05));" '
                     'alt="Departamento de Ingeniería Industrial - UdeC"/>')
else:
    logo_img_html = ""

header_html = f"""
<div style="display: flex; align-items: center; gap: 32px;
            padding: 18px 22px 14px 22px; margin-bottom: 14px;
            border-bottom: 2px solid #1A2530;">
    <div style="flex: 0 0 auto;">
        {logo_img_html}
    </div>
    <div style="flex: 1; min-width: 0;">
        <h1 style="margin: 0; padding: 0; line-height: 1.15;
                   font-size: 1.75rem; color: #1A2530;
                   font-family: 'Inter', 'Roboto', sans-serif;
                   font-weight: 700;">
            Distribución territorial de la pobreza multidimensional en Chile
        </h1>
        <p style="margin: 6px 0 0 0; font-size: 0.95rem; color: #2C3E50;
                  font-family: 'Inter', 'Roboto', sans-serif;">
            Análisis del origen de los hogares y la autocorrelación espacial
            de la pobreza · CASEN 2013–2024
        </p>
        <p style="margin: 4px 0 0 0; font-size: 0.82rem; color: #7F8C8D;
                  font-family: 'Inter', 'Roboto', sans-serif;">
            Grupo 11 · Matías Arriagada, José Luis Erices, Rayen Muñoz ·
            Data Visualization 2026-1
        </p>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# --- Fila 1: Gráficos A y B ---
col_a, col_b = st.columns(2)
with col_a:
    fig_donut = plot_a4_g1_donut(df_g1)
    st.plotly_chart(fig_donut, use_container_width=True)
with col_b:
    fig_macro = plot_a4_g2_macrozonas(df_g2_macro)
    st.plotly_chart(fig_macro, use_container_width=True)

# --- Fila 2: Mapas de reemplazo de B (ahora Gráfico C) ---
st.markdown("### **C. Tipo de pobreza más sobrerrepresentado por región**")
st.markdown("<p style='font-size: 0.95em; color: #2C3E50; margin-top: -0.5rem; margin-bottom: 1rem;'>Cada región se colorea según el tipo de pobreza cuya proporción supera en mayor medida el promedio nacional.</p>", unsafe_allow_html=True)

# Leyenda de escala de colores
col_leg1, col_leg2, col_leg3 = st.columns(3)
with col_leg1:
    st.markdown(
        """
        <div style="border-left: 10px solid #F39D2B; padding-left: 10px; background-color: #FAFAFA; border-radius: 4px; padding-top: 8px; padding-bottom: 8px;">
            <span style="font-weight: bold; color: #F39D2B; font-size: 1.05em;">Pobreza por ingresos</span><br>
            <span style="font-size: 0.85em; color: #2C3E50;">
                Color naranja sólido.
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )
with col_leg2:
    st.markdown(
        """
        <div style="border-left: 10px solid #E64F42; padding-left: 10px; background-color: #FAFAFA; border-radius: 4px; padding-top: 8px; padding-bottom: 8px;">
            <span style="font-weight: bold; color: #E64F42; font-size: 1.05em;">Pobreza multidimensional</span><br>
            <span style="font-size: 0.85em; color: #2C3E50;">
                Color rojo/coral sólido.
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )
with col_leg3:
    st.markdown(
        """
        <div style="border-left: 10px solid #731819; padding-left: 10px; background-color: #FAFAFA; border-radius: 4px; padding-top: 8px; padding-bottom: 8px;">
            <span style="font-weight: bold; color: #731819; font-size: 1.05em;">Pobreza por ingresos y multidimensional</span><br>
            <span style="font-size: 0.85em; color: #2C3E50;">
                Color burdeo sólido.
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

st.write("") # Espaciador

col_map1, col_map2, col_map3, col_map4, col_map5 = st.columns(5)
with col_map1:
    st.plotly_chart(plot_zone_map("Norte Grande", gdf_chile, df_regional), use_container_width=True)
with col_map2:
    st.plotly_chart(plot_zone_map("Norte Chico", gdf_chile, df_regional), use_container_width=True)
with col_map3:
    st.plotly_chart(plot_zone_map("Zona Central", gdf_chile, df_regional), use_container_width=True)
with col_map4:
    st.plotly_chart(plot_zone_map("Zona Sur", gdf_chile, df_regional), use_container_width=True)
with col_map5:
    st.plotly_chart(plot_zone_map("Zona Austral", gdf_chile, df_regional), use_container_width=True)

# --- Fila 3: Gráficos D y E ---
col_d, col_e = st.columns(2)
with col_d:
    fig_sankey = plot_a4_g10_sankey(df_g3)
    st.plotly_chart(fig_sankey, use_container_width=True)
with col_e:
    fig_barras = plot_a4_g3_barras(df_g3)
    st.plotly_chart(fig_barras, use_container_width=True)

# --- Fila 4: Gráficos F y G ---
col_f, col_g = st.columns(2)
with col_f:
    fig_radar = plot_a4_g4_radar(df_g4)
    st.plotly_chart(fig_radar, use_container_width=True)
with col_g:
    fig_lineas = plot_a4_g5_lineas(df_g5)
    st.plotly_chart(fig_lineas, use_container_width=True)

# --- Fila 5: Moran y LISA ---
col_moran, col_lisa = st.columns(2)
with col_moran:
    fig_moran = plot_moran_scatterplot(df_spatial, stats_moran)
    st.plotly_chart(fig_moran, use_container_width=True)
with col_lisa:
    fig_lisa = plot_lisa_comunal_norte_grande(gdf_lisa_ng)
    st.plotly_chart(fig_lisa, use_container_width=True)


# --- Footer institucional ----------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 2.4rem; padding: 14px 22px 10px 22px;
                border-top: 1px solid #ECF0F1;
                font-family: 'Inter', 'Roboto', sans-serif;
                font-size: 0.78rem; color: #7F8C8D;
                display: flex; justify-content: space-between;
                flex-wrap: wrap; gap: 8px;">
        <span><b>Fuentes:</b>
            Ministerio de Desarrollo Social y Familia · Encuestas CASEN
            2013, 2015, 2017, 2022, 2024 · Biblioteca del Congreso
            Nacional de Chile (cartografía regional).</span>
        <span>Universidad de Concepción · Dpto. Ingeniería Industrial ·
              Data Visualization 2026-1</span>
    </div>
    """,
    unsafe_allow_html=True
)
