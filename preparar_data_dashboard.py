# =============================================================================
#  preparar_data_dashboard.py  ·  Tarea 3 Data Visualization
# -----------------------------------------------------------------------------
#  Genera la carpeta liviana `data_dashboard/` que consume el dashboard de
#  Streamlit, leyendo las bases CASEN pesadas SOLO en local. El repositorio
#  de GitHub no necesita las bases completas: basta con `data_dashboard/`.
#
#  Bases pesadas de entrada (NO se suben al repo):
#    - zip/casen_2024.parquet                  (~46 MB, microdatos 2024)
#    - zip/casen_2024_provincia_comuna.dta     (provincia/comuna 2024)
#    - Regional.geojson                        (geometría regional, 1.4 MB)
#    - processed_data/H_lisa_norte_grande.geojson  (LISA comunal ya calculado)
#    - processed_data/g5_historical_inmig.csv  (serie histórica ya agregada)
#
#  Salidas livianas (carpeta data_dashboard/, una por gráfico):
#    grafico_A_distribucion_nacional.csv
#    grafico_B_macrozonas.csv
#    grafico_C_sobrerrepresentacion.csv
#    grafico_D_sankey_norte_grande.csv
#    grafico_E_origen_norte_grande.csv
#    grafico_F_dumbbell_carencias.csv
#    grafico_G_evolucion_inmigrantes.csv
#    grafico_H_lisa_comunal.csv
#    regiones_simplificado.geojson
#    comunas_norte_grande_simplificado.geojson
#
#  Uso:   python preparar_data_dashboard.py
# =============================================================================
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

# --- Rutas -------------------------------------------------------------------
DIR = Path(__file__).resolve().parent
DIR_OUT = DIR / "data_dashboard"
DIR_OUT.mkdir(exist_ok=True)

PARQUET_2024 = DIR / "zip" / "casen_2024.parquet"
REGIONAL_GEOJSON = DIR / "Regional.geojson"
# Insumos ya calculados que se reutilizan (LISA comunal y serie histórica):
# regenerarlos exige bases enormes (shapefile de 62 MB + 5 .dta de varios GB),
# por lo que se parte de los agregados ya validados en processed_data/.
LISA_GEOJSON_FUENTE = DIR / "processed_data" / "H_lisa_norte_grande.geojson"
G5_FUENTE = DIR / "processed_data" / "g5_historical_inmig.csv"

# Simplificación de geometrías. Chile trae ~10.000 polígonos (miles de islas
# minúsculas del sur austral) que inflan el GeoJSON sin aportar nada a la
# escala del dashboard. Se descartan los polígonos bajo `MIN_AREA_*` (grados²),
# se simplifica con `TOL_*` (grados) y se recortan decimales con
# `PREC_*` (COORDINATE_PRECISION de pyogrio). Resultado visualmente idéntico.
TOL_REGIONES = 0.012
MIN_AREA_REGIONES = 0.002
PREC_REGIONES = 4
TOL_COMUNAS = 0.004
MIN_AREA_COMUNAS = 0.0008
PREC_COMUNAS = 5

# --- Diccionarios de mapeo (idénticos a preprocess.py) ----------------------
REGION_A_ZONA = {
    15: "Norte Grande", 1: "Norte Grande", 2: "Norte Grande",
    3: "Norte Chico", 4: "Norte Chico",
    5: "Zona Central", 13: "Zona Central", 6: "Zona Central", 7: "Zona Central",
    16: "Zona Sur", 8: "Zona Sur", 9: "Zona Sur", 14: "Zona Sur", 10: "Zona Sur",
    11: "Zona Austral", 12: "Zona Austral",
}
DICC_REGIONES = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "La Araucanía",
    14: "Los Ríos", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
}
DIMENSIONES_MD_2015 = {
    "Educacion":             ["hh_d_asis_2015", "hh_d_rez_2015", "hh_d_esc_2015"],
    "Salud":                 ["hh_d_mal_2015", "hh_d_prevs_2015", "hh_d_acc_2015"],
    "Trabajo y Seg. Social": ["hh_d_act_2015", "hh_d_cot_2015", "hh_d_jub_2015"],
    "Vivienda y Entorno":    ["hh_d_hacina_2015", "hh_d_estado_2015",
                              "hh_d_servbas_2015", "hh_d_entorno_2015"],
    "Redes y Cohesion":      ["hh_d_appart_2015", "hh_d_tsocial_2015", "hh_d_seg_2015"],
}
DIMENSIONES_MD_2024 = {
    "Educación": ["hh_d_asis", "hh_d_rez", "hh_d_esc", "hh_d_ape"],
    "Salud": ["hh_d_acc", "hh_d_ali", "hh_d_contprev", "hh_d_dpf"],
    "Trabajo y Seguridad Social":
        ["hh_d_actsub", "hh_d_inf", "hh_d_jub", "hh_d_cui"],
    "Vivienda y Entorno":
        ["hh_d_defcuanti", "hh_d_defcuali", "hh_d_accesi", "hh_d_medio"],
    "Redes y Cohesión Social":
        ["hh_d_apoyo", "hh_d_tsocial", "hh_d_seg", "hh_d_conec"],
}


def _descartar_islas(geom, min_area):
    """Conserva solo los polígonos de un (Multi)Polygon con área >= min_area
    (en grados²). Siempre deja al menos el polígono más grande para no
    perder la región/comuna entera."""
    from shapely.geometry import MultiPolygon
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return geom
    grandes = [p for p in geom.geoms if p.area >= min_area]
    if not grandes:
        grandes = [max(geom.geoms, key=lambda p: p.area)]
    return MultiPolygon(grandes) if len(grandes) > 1 else grandes[0]


def clasificar_pobreza(es_pob_ing, es_pob_multi):
    if es_pob_ing and es_pob_multi:
        return "Pobreza ingresos y multidim."
    if es_pob_ing:
        return "Pobreza por ingresos"
    if es_pob_multi:
        return "Pobreza multidimensional"
    return "Fuera de pobreza"


# --- Dimensiones de filtro (drill-down) -------------------------------------
# Columnas categóricas del jefe de hogar que se guardan en los cubos para
# permitir filtrado/recalculo real en el dashboard. Niveles fijos y pocos,
# de modo que los cubos siguen pesando KB.
DIMS_FILTRO = ["area_f", "sexo_jefe", "edad_tramo", "quintil"]


def agregar_dims_filtro(df_h):
    """Agrega al DataFrame de jefes las 4 dimensiones de filtro. Los nulos
    se etiquetan 'Sin dato' para que la suma del cubo reproduzca el total
    (y el filtro 'Todas/Todos' sea exacto)."""
    df = df_h.copy()
    df["area_f"] = df["area"].map({1: "Urbano", 2: "Rural"}).fillna("Sin dato")
    df["sexo_jefe"] = df["sexo"].map({1: "Hombre", 2: "Mujer"}).fillna("Sin dato")
    df["edad_tramo"] = pd.cut(
        pd.to_numeric(df["edad"], errors="coerce"),
        bins=[-np.inf, 29, 59, np.inf],
        labels=["Joven (≤29)", "Adulto (30-59)", "Mayor (60+)"]
    ).astype("object").fillna("Sin dato")
    df["quintil"] = df["qaut"].map(
        {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}).fillna("Sin dato")
    return df


# =============================================================================
# 1. CARGA DE LA BASE 2024 (parquet) Y CONSTRUCCION DE df_h (jefes de hogar)
# =============================================================================
def cargar_hogares_2024():
    print(f"Leyendo {PARQUET_2024.name} ...")
    cols_base = ["folio", "pco1_a", "expr", "region", "area", "pobreza",
                 "pobreza_multi", "lugar_nac", "estrato",
                 "sexo", "edad", "qaut"]  # 3 últimas: dimensiones de filtro
    cols = list(cols_base)
    for dv in DIMENSIONES_MD_2015.values():
        cols += dv
    for dv in DIMENSIONES_MD_2024.values():
        cols += dv
    cols = list(dict.fromkeys(cols))  # dedup preservando orden

    df = pd.read_parquet(PARQUET_2024, columns=cols)
    # Un registro por hogar (jefe)
    df_h = df[df["pco1_a"] == 1].drop_duplicates(subset=["folio"]).copy()
    for c in cols_base:
        df_h[c] = pd.to_numeric(df_h[c], errors="coerce")
    df_h = df_h.dropna(subset=["expr"])
    # Excluir islas de la Zona Central (Juan Fernández=5104, Isla de Pascua=5201)
    df_h = df_h[~((df_h["estrato"] // 100).isin([5104, 5201]))]

    df_h["es_pob_ing"] = df_h["pobreza"].isin([1, 2]).astype(int)
    df_h["es_pob_multi"] = (df_h["pobreza_multi"] == 1).astype(int)
    df_h["estado_pob"] = df_h.apply(
        lambda r: clasificar_pobreza(r["es_pob_ing"], r["es_pob_multi"]), axis=1)
    df_h["es_pobre"] = (df_h["estado_pob"] != "Fuera de pobreza").astype(int)
    df_h["zona"] = df_h["region"].map(REGION_A_ZONA)
    df_h["origen_jefe"] = df_h["lugar_nac"].map(
        {0: "Hogares chilenos", 1: "Hogares inmigrantes"})
    df_h = agregar_dims_filtro(df_h)
    print(f"  {len(df_h):,} hogares (jefes) tras limpieza.")
    return df_h


# =============================================================================
# 2. GRAFICOS A–F (todos derivados del parquet 2024)
# =============================================================================
def graf_A(df_h):
    # Cubo nacional: estado_pob x dims de filtro. La app suma sobre las dims
    # no seleccionadas para recomputar el donut con drill-down real.
    print("A · Distribución nacional (cubo) ...")
    agg = df_h.groupby(["estado_pob"] + DIMS_FILTRO, observed=True).agg(
        expr=("expr", "sum"), n=("expr", "count")).reset_index()
    agg = agg[agg["expr"] > 0]
    agg.to_csv(DIR_OUT / "grafico_A_distribucion_nacional.csv", index=False)


def graf_B(df_h):
    # Cubo zonal: zona x estado_pob x dims (solo hogares pobres).
    print("B · Composición por macrozona (cubo) ...")
    dfp = df_h[(df_h["es_pobre"] == 1) & df_h["zona"].notna()].copy()
    agg = dfp.groupby(["zona", "estado_pob"] + DIMS_FILTRO, observed=True).agg(
        expr=("expr", "sum"), n=("expr", "count")).reset_index()
    agg = agg[agg["expr"] > 0]
    agg.to_csv(DIR_OUT / "grafico_B_macrozonas.csv", index=False)


def graf_C(df_h):
    # Cubo regional: region x estado_pob x dims, TODOS los hogares (incluye
    # 'Fuera de pobreza'). La app recomputa %tipos entre pobres, predominante,
    # sobrerrepresentación vs nacional y tamaños muestrales según el filtro.
    print("C · Sobrerrepresentación regional (cubo) ...")
    agg = df_h.groupby(["region", "estado_pob"] + DIMS_FILTRO,
                       observed=True).agg(
        expr=("expr", "sum"), n=("expr", "count")).reset_index()
    agg = agg[agg["expr"] > 0]
    agg["region"] = agg["region"].astype(int)
    agg["region_name"] = agg["region"].map(DICC_REGIONES)
    agg["zona"] = agg["region"].map(REGION_A_ZONA)
    agg.to_csv(DIR_OUT / "grafico_C_sobrerrepresentacion.csv", index=False)


def _trifecta(row):
    if row["total_car_20"] == 0:
        return "Sin carencias"
    if row.get("hh_d_defcuanti") == 1: return "Déficit cuantitativo"
    if row.get("hh_d_esc") == 1:        return "Escolaridad"
    if row.get("hh_d_conec") == 1:      return "Conectividad digital"
    if row.get("hh_d_inf") == 1:        return "Informalidad"
    return "Otras carencias"


def graf_D_E(df_h):
    print("D y E · Norte Grande (Sankey y barras por origen) ...")
    df_pobres = df_h[df_h["es_pobre"] == 1].copy()
    ng = df_pobres[(df_pobres["zona"] == "Norte Grande")
                   & df_pobres["origen_jefe"].notna()].copy()
    cols_20 = [c for inds in DIMENSIONES_MD_2024.values() for c in inds]
    for c in cols_20:
        ng[c] = pd.to_numeric(ng[c], errors="coerce").fillna(0).astype(int)
    ng["total_car_20"] = ng[cols_20].sum(axis=1)
    ng["trifecta"] = ng.apply(_trifecta, axis=1)

    # D: flujos estado -> origen -> trifecta. El Sankey no se filtra por las
    # dimensiones demográficas (universo fijo), así que se mantiene agregado.
    agg_d = ng.groupby(["estado_pob", "origen_jefe", "trifecta"])["expr"].sum().reset_index()
    agg_d.to_csv(DIR_OUT / "grafico_D_sankey_norte_grande.csv", index=False)

    # E: cubo origen x estado x dims (Norte Grande, solo pobres) para
    # drill-down real en las barras por origen.
    agg_e = ng.groupby(["origen_jefe", "estado_pob"] + DIMS_FILTRO,
                       observed=True).agg(
        expr=("expr", "sum"), n=("expr", "count")).reset_index()
    agg_e = agg_e[agg_e["expr"] > 0]
    agg_e.to_csv(DIR_OUT / "grafico_E_origen_norte_grande.csv", index=False)


def graf_F(df_h):
    # Cubo dumbbell: origen x dims. Para recomputar el promedio ponderado de
    # carencias por dimensión sobre cualquier subconjunto, se guarda por combo
    # la suma de pesos (expr) y la suma ponderada de cada score (w_<dim> =
    # Σ score·expr). En la app: promedio = Σw_<dim> / Σexpr.
    print("F · Dumbbell de carencias (cubo) ...")
    ng = df_h[(df_h["zona"] == "Norte Grande") & df_h["origen_jefe"].notna()].copy()
    for dv in DIMENSIONES_MD_2015.values():
        for col in dv:
            ng[col] = pd.to_numeric(ng[col], errors="coerce").fillna(0)
    for dim, inds in DIMENSIONES_MD_2015.items():
        ng[f"w_{dim}"] = ng[inds].sum(axis=1) * ng["expr"]  # score·peso

    aggs = {"expr": ("expr", "sum"), "n": ("expr", "count")}
    for dim in DIMENSIONES_MD_2015:
        aggs[f"w_{dim}"] = (f"w_{dim}", "sum")
    cubo = ng.groupby(["origen_jefe"] + DIMS_FILTRO,
                      observed=True).agg(**aggs).reset_index()
    cubo = cubo[cubo["expr"] > 0]
    cubo.to_csv(DIR_OUT / "grafico_F_dumbbell_carencias.csv", index=False)


# =============================================================================
# 3. GRAFICO G (serie histórica) — se reutiliza el agregado ya validado
# =============================================================================
def graf_G():
    print("G · Evolución histórica de inmigrantes (reusa agregado) ...")
    # Regenerarlo exige los 5 .dta históricos (cientos de MB a 1.5 GB c/u).
    # El agregado ya está validado (25 filas: 5 zonas x 5 olas).
    df = pd.read_csv(G5_FUENTE).drop_duplicates(subset=["anio", "zona"])
    df.to_csv(DIR_OUT / "grafico_G_evolucion_inmigrantes.csv", index=False)


# =============================================================================
# 4. GRAFICO H — separar el GeoJSON pesado en CSV liviano + geometría simple
# =============================================================================
def graf_H():
    print("H · LISA comunal: separar atributos (CSV) y geometría (geojson) ...")
    gdf = gpd.read_file(LISA_GEOJSON_FUENTE)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Atributos (sin geometría) -> CSV liviano, unido luego por comuna_norm
    cols_attr = ["comuna_norm", "Comuna", "pct_pobreza", "pobre_pond",
                 "total_pond", "n_sample", "n_sample_poor", "lisa_cat",
                 "lisa_p_value", "moran_quadrant"]
    cols_attr = [c for c in cols_attr if c in gdf.columns]
    df_attr = pd.DataFrame(gdf.drop(columns="geometry"))[cols_attr]
    df_attr.to_csv(DIR_OUT / "grafico_H_lisa_comunal.csv", index=False)

    # Geometría simplificada -> geojson (solo clave de unión + forma)
    geo = gdf[["comuna_norm", "geometry"]].copy()
    geo["geometry"] = geo["geometry"].apply(
        lambda g: _descartar_islas(g, MIN_AREA_COMUNAS))
    geo["geometry"] = geo["geometry"].simplify(TOL_COMUNAS,
                                               preserve_topology=True)
    geo.to_file(DIR_OUT / "comunas_norte_grande_simplificado.geojson",
                driver="GeoJSON", COORDINATE_PRECISION=PREC_COMUNAS)


# =============================================================================
# 5. GEOMETRIA REGIONAL simplificada (mapas C)
# =============================================================================
def geom_regiones():
    print("Geometría regional simplificada ...")
    gdf = gpd.read_file(REGIONAL_GEOJSON)
    gdf["codregion"] = gdf["codregion"].astype(int)
    gdf = gdf[gdf["codregion"] != 0].copy()
    # Recortar islas de Valparaíso (código 5) para que el mapa quede continental
    from shapely.geometry import MultiPolygon
    idx5 = gdf[gdf["codregion"] == 5].index
    if not idx5.empty:
        geom = gdf.loc[idx5[0], "geometry"]
        if isinstance(geom, MultiPolygon):
            cont = [p for p in geom.geoms if p.bounds[0] > -73.0]
            gdf.loc[idx5[0], "geometry"] = MultiPolygon(cont)
    gdf = gdf[["codregion", "geometry"]].copy()
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: _descartar_islas(g, MIN_AREA_REGIONES))
    gdf["geometry"] = gdf["geometry"].simplify(TOL_REGIONES,
                                               preserve_topology=True)
    gdf.to_file(DIR_OUT / "regiones_simplificado.geojson",
                driver="GeoJSON", COORDINATE_PRECISION=PREC_REGIONES)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Preparando data_dashboard/ (insumos livianos para Streamlit)")
    print("=" * 70)
    df_h = cargar_hogares_2024()
    graf_A(df_h)
    graf_B(df_h)
    graf_C(df_h)
    graf_D_E(df_h)
    graf_F(df_h)
    graf_G()
    graf_H()
    geom_regiones()

    print("\nArchivos generados en data_dashboard/:")
    total = 0
    for f in sorted(DIR_OUT.iterdir()):
        kb = f.stat().st_size / 1024
        total += kb
        print(f"  {kb:8.1f} KB  {f.name}")
    print(f"  {'-'*40}\n  {total/1024:8.2f} MB  TOTAL")
    print("\nListo. El dashboard puede correr solo con data_dashboard/.")


if __name__ == "__main__":
    main()
