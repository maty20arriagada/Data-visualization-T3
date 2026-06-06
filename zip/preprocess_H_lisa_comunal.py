# =============================================================================
#  Pre-procesamiento del Mapa LISA comunal del Norte Grande
#  (pobreza por ingresos Y multidimensional simultaneamente)
#
#  Replica EXACTAMENTE el calculo del script
#  exportar_H_ingresos_y_multidim.py (paleta, ponderacion expc, lista de 20
#  comunas, semilla 42, Queen->KNN4) y guarda el resultado como GeoJSON
#  reproyectado a EPSG:4326 listo para Plotly Choropleth en el dashboard.
#
#  Ejecutar UNA SOLA VEZ:
#     cd zip; python preprocess_H_lisa_comunal.py
#  Salida: ../processed_data/H_lisa_norte_grande.geojson
# =============================================================================
from pathlib import Path
import unicodedata
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal as lps
from esda.moran import Moran_Local

warnings.filterwarnings("ignore")


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).upper()
    text = (unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore").decode("utf-8"))
    return text.strip()


DIR_ZIP = Path(__file__).resolve().parent
DIR_PROC = DIR_ZIP.parent / "processed_data"
DIR_PROC.mkdir(parents=True, exist_ok=True)

print("Cargando CASEN 2024 (parquet) ...")
df_pq = pd.read_parquet(
    DIR_ZIP / "casen_2024.parquet",
    columns=["folio", "id_persona", "pobreza", "pobreza_multi_2015"],
)
df_dta = pd.read_stata(DIR_ZIP / "casen_2024_provincia_comuna.dta")

print("Cruzando y filtrando jefes de hogar ...")
df = pd.merge(df_pq, df_dta, on=["folio", "id_persona"], how="inner")
df = df[df["id_persona"] == 1].copy()
df["comuna_norm"] = df["comuna"].apply(normalize_text)
df["pobreza"] = pd.to_numeric(df["pobreza"], errors="coerce")
df["pobreza_multi_2015"] = pd.to_numeric(df["pobreza_multi_2015"], errors="coerce")
df["es_pobre_ingreso"] = df["pobreza"].isin([1, 2]).astype(int)
df["es_pobre_multi"] = (df["pobreza_multi_2015"] == 1).astype(int)
df["pobreza_simultanea"] = ((df["es_pobre_ingreso"] == 1)
                             & (df["es_pobre_multi"] == 1)).astype(int)
df["pobre_pond"] = df["pobreza_simultanea"] * df["expc"]

comunal = (df.groupby("comuna_norm")
           .agg(pobre_pond=("pobre_pond", "sum"),
                total_pond=("expc", "sum"),
                n_sample=("folio", "count"),
                n_sample_poor=("pobreza_simultanea", "sum"))
           .reset_index())
comunal["pct_pobreza"] = (comunal["pobre_pond"]
                           / comunal["total_pond"]) * 100

print("Cargando shapefile de comunas ...")
gdf = gpd.read_file(DIR_ZIP / "comunas" / "comunas.shp")
gdf["comuna_norm"] = gdf["Comuna"].apply(normalize_text)

# Lista canonica del Norte Grande (3 regiones: Arica y Parin., Tarapaca,
# Antofagasta). Identica a la del script exportar_H_ingresos_y_multidim.py.
NORTE_GRANDE = [
    "ARICA", "CAMARONES", "PUTRE", "GENERAL LAGOS",
    "IQUIQUE", "ALTO HOSPICIO", "POZO ALMONTE", "CAMINA", "COLCHANE",
    "HUARA", "PICA",
    "ANTOFAGASTA", "MEJILLONES", "SIERRA GORDA", "TALTAL", "CALAMA",
    "OLLAGUE", "SAN PEDRO DE ATACAMA", "TOCOPILLA", "MARIA ELENA",
]
gdf_ng = gdf[gdf["comuna_norm"].isin(NORTE_GRANDE)].copy()
gdf_ng = gdf_ng.merge(comunal, on="comuna_norm", how="left")

gdf_sin_datos = gdf_ng[gdf_ng["pct_pobreza"].isna()].copy()
gdf_datos = gdf_ng[~gdf_ng["pct_pobreza"].isna()].copy()

print(f"Comunas con datos: {len(gdf_datos)} | "
      f"Comunas sin datos: {len(gdf_sin_datos)} "
      f"({gdf_sin_datos['comuna_norm'].tolist()})")

# --- LISA: matriz de pesos espaciales --------------------------------------
print("Calculando LISA (Moran Local) ...")
w = lps.weights.Queen.from_dataframe(gdf_datos)
if w.islands:
    print(f"Queen tiene islas {w.islands}. Cambiando a KNN=4.")
    w = lps.weights.KNN.from_dataframe(gdf_datos, k=4)
else:
    print("Contiguidad Queen sin islas.")
w.transform = "r"

y = gdf_datos["pct_pobreza"].values
moran_loc = Moran_Local(y, w, seed=42)

sig = 0.05
hotspots = (moran_loc.p_sim <= sig) & (moran_loc.q == 1)
coldspots = (moran_loc.p_sim <= sig) & (moran_loc.q == 3)
doughnuts = (moran_loc.p_sim <= sig) & (moran_loc.q == 2)
diamonds = (moran_loc.p_sim <= sig) & (moran_loc.q == 4)

gdf_datos["lisa_cat"] = "No significativo"
gdf_datos.loc[hotspots, "lisa_cat"] = "Alto-Alto"
gdf_datos.loc[coldspots, "lisa_cat"] = "Bajo-Bajo"
gdf_datos.loc[doughnuts, "lisa_cat"] = "Bajo-Alto"
gdf_datos.loc[diamonds, "lisa_cat"] = "Alto-Bajo"
gdf_datos["lisa_p_value"] = moran_loc.p_sim
gdf_datos["moran_quadrant"] = moran_loc.q

# Comunas sin datos: tampoco tienen LISA
gdf_sin_datos["lisa_cat"] = "Sin datos"
gdf_sin_datos["lisa_p_value"] = np.nan
gdf_sin_datos["moran_quadrant"] = np.nan
for c in ["pobre_pond", "total_pond", "pct_pobreza", "n_sample",
          "n_sample_poor"]:
    if c not in gdf_sin_datos.columns:
        gdf_sin_datos[c] = np.nan

# Union final, proyeccion para Plotly (EPSG:4326 = lon/lat)
gdf_final = pd.concat([gdf_datos, gdf_sin_datos], ignore_index=True)
gdf_final = gpd.GeoDataFrame(gdf_final, geometry="geometry",
                              crs=gdf.crs).to_crs(epsg=4326)

# Selecciona solo columnas relevantes para el dashboard
cols_out = ["Comuna", "comuna_norm", "pct_pobreza", "pobre_pond",
            "total_pond", "n_sample", "n_sample_poor", "lisa_cat",
            "lisa_p_value", "moran_quadrant", "geometry"]
cols_out = [c for c in cols_out if c in gdf_final.columns]
gdf_final = gdf_final[cols_out]

out_geojson = DIR_PROC / "H_lisa_norte_grande.geojson"
gdf_final.to_file(out_geojson, driver="GeoJSON")
print(f"\nGuardado: {out_geojson}")
print("\nResumen final:")
print(gdf_final[["Comuna", "pct_pobreza", "lisa_cat", "n_sample"]]
      .sort_values("pct_pobreza", ascending=False, na_position="last")
      .to_string(index=False))
