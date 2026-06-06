import pandas as pd
import geopandas as gpd
import unicodedata
import libpysal as lps
from esda.moran import Moran_Local
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def normalize_text(text):
    if pd.isna(text): return ''
    text = str(text).upper()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.strip()

DIR_OUT = Path("outputs_graficos")
DIR_OUT.mkdir(exist_ok=True)

print("Cargando datos...")

# 1. Cargar datos (incluyendo la variable 'pobreza' para ingresos)
df_pq = pd.read_parquet('casen_2024.parquet', columns=['folio', 'id_persona', 'pobreza', 'pobreza_multi_2015'])
df_dta = pd.read_stata('casen_2024_provincia_comuna.dta')

# 2. Merge y preparar variables
df = pd.merge(df_pq, df_dta, on=['folio', 'id_persona'], how='inner')
df = df[df['id_persona'] == 1].copy()

df['comuna_norm'] = df['comuna'].apply(normalize_text)

# Limpiar variables
df['pobreza'] = pd.to_numeric(df['pobreza'], errors='coerce')
df['pobreza_multi_2015'] = pd.to_numeric(df['pobreza_multi_2015'], errors='coerce')

# Condición: Pobreza por ingresos (1=extremo, 2=no extremo) Y pobreza multidimensional (1=pobre)
df['es_pobre_ingreso'] = df['pobreza'].isin([1, 2]).astype(int)
df['es_pobre_multi'] = (df['pobreza_multi_2015'] == 1).astype(int)
df['pobreza_simultanea'] = ((df['es_pobre_ingreso'] == 1) & (df['es_pobre_multi'] == 1)).astype(int)

df['pobre_pond'] = df['pobreza_simultanea'] * df['expc']

comunal = df.groupby('comuna_norm').agg(
    pobre_pond=('pobre_pond', 'sum'),
    total_pond=('expc', 'sum')
).reset_index()

comunal['pct_pobreza'] = (comunal['pobre_pond'] / comunal['total_pond']) * 100

# 3. Cargar shapefile
print("Cargando geometrías...")
gdf = gpd.read_file('comunas/comunas.shp')
gdf['comuna_norm'] = gdf['Comuna'].apply(normalize_text)

norte_grande = [
    'ARICA', 'CAMARONES', 'PUTRE', 'GENERAL LAGOS',
    'IQUIQUE', 'ALTO HOSPICIO', 'POZO ALMONTE', 'CAMINA', 'COLCHANE', 'HUARA', 'PICA',
    'ANTOFAGASTA', 'MEJILLONES', 'SIERRA GORDA', 'TALTAL', 'CALAMA', 'OLLAGUE', 'SAN PEDRO DE ATACAMA', 'TOCOPILLA', 'MARIA ELENA'
]

gdf_ng = gdf[gdf['comuna_norm'].isin(norte_grande)].copy()
gdf_ng = gdf_ng.merge(comunal, on='comuna_norm', how='left')

gdf_sin_datos = gdf_ng[gdf_ng['pct_pobreza'].isna()].copy()
gdf_datos = gdf_ng[~gdf_ng['pct_pobreza'].isna()].copy()

print(f"Comunas sin datos: {gdf_sin_datos['comuna_norm'].tolist()}")

# 5. Spatial weights
print("Calculando LISA...")
w = lps.weights.Queen.from_dataframe(gdf_datos)
if w.islands:
    print(f"Queen tiene islas {w.islands}. Usando KNN=4.")
    w = lps.weights.KNN.from_dataframe(gdf_datos, k=4)
else:
    print("Usando contigüidad Queen.")

w.transform = 'r'

y = gdf_datos['pct_pobreza'].values
moran_loc = Moran_Local(y, w, seed=42)

sig = 0.05
hotspots = (moran_loc.p_sim <= sig) & (moran_loc.q == 1)
coldspots = (moran_loc.p_sim <= sig) & (moran_loc.q == 3)
doughnuts = (moran_loc.p_sim <= sig) & (moran_loc.q == 2)
diamonds = (moran_loc.p_sim <= sig) & (moran_loc.q == 4)

gdf_datos['lisa_cat'] = "No significativo"
gdf_datos.loc[hotspots, 'lisa_cat'] = "Alto-Alto"
gdf_datos.loc[coldspots, 'lisa_cat'] = "Bajo-Bajo"
gdf_datos.loc[doughnuts, 'lisa_cat'] = "Bajo-Alto"
gdf_datos.loc[diamonds, 'lisa_cat'] = "Alto-Bajo"

gdf_sin_datos['lisa_cat'] = "Sin datos"
gdf_final = pd.concat([gdf_datos, gdf_sin_datos])

print("Generando gráfico...")
paleta = {
    "Alto-Alto": "#731819", 
    "Bajo-Bajo": "#A64B5A", 
    "Alto-Bajo": "#C07884", 
    "Bajo-Alto": "#E4B7BB", 
    "No significativo": "#D9D9D9", 
    "Sin datos": "#F8F9F9"
}

# ----------------- AJUSTES VISUALES: MAPA LISA PURO -----------------
fig = plt.figure(figsize=(9, 13))
ax = fig.add_axes([0.1, 0.15, 0.8, 0.75])

for ctype, data in gdf_final.groupby('lisa_cat'):
    # Tramado para sin datos
    if ctype == 'Sin datos':
        data.plot(ax=ax, color=paleta[ctype], edgecolor='gray', linewidth=0.5, hatch='///')
    else:
        # Borde blanco para separar comunas limpiamente
        data.plot(ax=ax, color=paleta[ctype], edgecolor='white', linewidth=0.5)

ax.set_axis_off()

# Títulos
plt.figtext(0.1, 0.95, "H. Clústeres espaciales de pobreza por ingresos y multidimensional", fontsize=15, fontweight='bold', color="#2C3E50", ha='left')
plt.figtext(0.1, 0.92, "Mapa LISA comunal en Norte Grande.", fontsize=13, color="#34495E", ha='left')

# Leyenda categórica
cat_ordenadas = ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto", "No significativo", "Sin datos"]
cat_presentes = [c for c in cat_ordenadas if c in gdf_final['lisa_cat'].unique()]

patches = []
for c in cat_presentes:
    if c == 'Sin datos':
        patches.append(mpatches.Patch(facecolor=paleta[c], edgecolor='gray', hatch='///', label=c))
    else:
        patches.append(mpatches.Patch(facecolor=paleta[c], edgecolor='black', linewidth=0.5, label=c))
        
ax.legend(
    handles=patches, 
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=False, 
    fontsize=11, 
    title="Clasificación LISA", 
    title_fontsize=12,
    columnspacing=1.5
)

# Nota
nota = "Nota: La desagregación comunal no es representativa en CASEN; sin embargo, permite observar un patrón territorial similar."
plt.figtext(0.5, 0.02, nota, fontsize=10, color="#7F8C8D", ha='center', wrap=True)

out_png = DIR_OUT / "grafico_H_lisa_comunal_norte_grande_ingresos_y_multidim.png"
out_pdf = DIR_OUT / "grafico_H_lisa_comunal_norte_grande_ingresos_y_multidim.pdf"
out_svg = DIR_OUT / "grafico_H_lisa_comunal_norte_grande_ingresos_y_multidim.svg"

plt.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
plt.savefig(out_pdf, facecolor='white', bbox_inches='tight')
plt.savefig(out_svg, facecolor='white', bbox_inches='tight')

print(f"Exportado exitosamente a {out_png}")
