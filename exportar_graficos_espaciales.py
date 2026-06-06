# %% 1. Carga de librerías, datos y configuración visual
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import libpysal
import esda

# Crear carpeta de salidas si no existe
DIR_APP = Path(__file__).resolve().parent
DIR_PROCESSED = DIR_APP / "processed_data"
GEOJSON_PATH = DIR_APP / "Regional.geojson"
DIR_OUT = DIR_APP / "outputs_graficos"
DIR_OUT.mkdir(exist_ok=True)

# Configuraciones visuales y constantes
FONT_FAMILY = "sans-serif"
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

COLOR_TEXTO_FUERTE = "#1A2530"
COLOR_TEXTO = "#2C3E50"

# Códigos regionales
CODIGOS_REGIONALES = {
    "Arica y Parinacota": "XV",
    "Tarapacá": "I",
    "Antofagasta": "II",
    "Atacama": "III",
    "Coquimbo": "IV",
    "Valparaíso": "V",
    "Metropolitana de Santiago": "RM",
    "Metropolitana": "RM",
    "Libertador General Bernardo O'Higgins": "VI",
    "O'Higgins": "VI",
    "Maule": "VII",
    "Ñuble": "XVI",
    "Biobío": "VIII",
    "La Araucanía": "IX",
    "Los Ríos": "XIV",
    "Los Lagos": "X",
    "Aysén del General Carlos Ibáñez del Campo": "XI",
    "Aysén": "XI",
    "Magallanes y de la Antártica Chilena": "XII",
    "Magallanes": "XII"
}

def get_cod_region(nombre):
    for k, v in CODIGOS_REGIONALES.items():
        if k in nombre:
            return v
    return nombre

MAPA_SIGLAS_REGIONALES = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
    9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "RM", 14: "XIV", 15: "XV", 16: "XVI"
}

# Carga de datos
print("Cargando datos...")
gdf_chile = gpd.read_file(GEOJSON_PATH)
gdf_chile["codregion"] = gdf_chile["codregion"].astype(int)
gdf_chile = gdf_chile[gdf_chile["codregion"] != 0].copy()

# Limpiar geometría de la Región de Valparaíso (código 5) para excluir islas
idx_reg5 = gdf_chile[gdf_chile["codregion"] == 5].index
if not idx_reg5.empty:
    from shapely.geometry import MultiPolygon
    geom = gdf_chile.loc[idx_reg5[0], "geometry"]
    if isinstance(geom, MultiPolygon):
        continental_polys = [p for p in geom.geoms if p.bounds[0] > -73.0]
        gdf_chile.loc[idx_reg5[0], "geometry"] = MultiPolygon(continental_polys)

df_regional = pd.read_csv(DIR_PROCESSED / "regional_poverty.csv")

# Porcentajes nacionales (calculados dinámicamente usando g1_donut)
df_donut = pd.read_csv(DIR_PROCESSED / "g1_donut.csv")
df_donut_pobres = df_donut[df_donut["estado_pob"] != "Fuera de pobreza"]
total_pobres = df_donut_pobres["expr"].sum()

nat_ingresos = round((df_donut_pobres[df_donut_pobres["estado_pob"] == "Pobreza por ingresos"]["expr"].values[0] / total_pobres) * 100, 2)
nat_multi = round((df_donut_pobres[df_donut_pobres["estado_pob"] == "Pobreza multidimensional"]["expr"].values[0] / total_pobres) * 100, 2)
nat_ambas = round((df_donut_pobres[df_donut_pobres["estado_pob"] == "Pobreza ingresos y multidim."]["expr"].values[0] / total_pobres) * 100, 2)

# Cálculo de la sobrerrepresentación
df_regional["diff_ingresos"] = df_regional["pct_pob_ingresos"] - nat_ingresos
df_regional["diff_multi"] = df_regional["pct_pob_multidimensional"] - nat_multi
df_regional["diff_ambas"] = df_regional["pct_pob_ambas"] - nat_ambas

def get_max_sobrerrepresentacion(row):
    diffs = {
        "Pobreza por ingresos": row["diff_ingresos"],
        "Pobreza multidimensional": row["diff_multi"],
        "Pobreza ingresos y multidim.": row["diff_ambas"]
    }
    cat_max = max(diffs, key=diffs.get)
    val_max = diffs[cat_max]
    return pd.Series([cat_max, val_max])

df_regional[["cat_sobrerep", "val_sobrerep"]] = df_regional.apply(get_max_sobrerrepresentacion, axis=1)

# Unir con geometrías
gdf_plot = gdf_chile.merge(df_regional, left_on="codregion", right_on="region")

# Carga de datos para Moran y LISA
df_spatial_raw = pd.read_csv(DIR_PROCESSED / "moran_lisa.csv")
# Estandarizar nombres para el merge
df_spatial_raw["region_name"] = df_spatial_raw["region_name"].str.replace("", "í").str.replace("Valparaso", "Valparaíso").str.replace("Tarapac", "Tarapacá")

gdf_spatial = gdf_chile.merge(df_spatial_raw[["region", "pobreza_multi_pct"]], left_on="codregion", right_on="region")

# Recalcular Moran y LISA local
y = gdf_spatial['pobreza_multi_pct'].values
w = libpysal.weights.Queen.from_dataframe(gdf_spatial, use_index=False)
w.transform = 'r'

moran = esda.Moran(y, w)
lisa = esda.Moran_Local(y, w)

y_std = (y - y.mean()) / y.std()
spatial_lag = libpysal.weights.lag_spatial(w, y_std)

gdf_spatial["y_std"] = y_std
gdf_spatial["spatial_lag"] = spatial_lag
gdf_spatial["lisa_q"] = lisa.q
gdf_spatial["lisa_p"] = lisa.p_sim

def classify_lisa(row):
    if row["lisa_p"] > 0.05:
        return "No significativo"
    else:
        if row["lisa_q"] == 1: return "Alto-Alto"
        elif row["lisa_q"] == 2: return "Bajo-Alto"
        elif row["lisa_q"] == 3: return "Bajo-Bajo"
        elif row["lisa_q"] == 4: return "Alto-Bajo"
    return "No significativo"

gdf_spatial["lisa_class"] = gdf_spatial.apply(classify_lisa, axis=1)
gdf_spatial["cod_region"] = gdf_spatial["region"].map(MAPA_SIGLAS_REGIONALES)

# %% 1.2. Gráfico A: Donut Nacional
print("Generando Gráfico A: Donut Nacional...")
orden_g1 = ["Fuera de pobreza", "Pobreza por ingresos", "Pobreza multidimensional", "Pobreza ingresos y multidim."]
df_donut_sorted = df_donut.set_index("estado_pob").reindex(orden_g1).reset_index()

values_a = df_donut_sorted["expr"].values
pcts_a = values_a / values_a.sum() * 100
total_pobres_pct = pcts_a[1:].sum()
n_muestral = int(df_donut_sorted["n"].sum()) if "n" in df_donut_sorted.columns else 78654
n_ponderado = float(df_donut_sorted["expr"].sum())

colors_a = ["#BDC3C7", "#F39D2B", "#E64F42", "#731819"]

fig_a, ax_a = plt.subplots(figsize=(8, 8))
wedges, texts = ax_a.pie(
    values_a, 
    colors=colors_a, 
    startangle=90, 
    counterclock=False, 
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=1.5),
    radius=1.0
)

ax_a.text(0, 0.1, f"{total_pobres_pct:.1f}%", fontsize=36, fontweight='bold', color="#78281F", ha='center', va='center')
ax_a.text(0, -0.15, "hogares en\nalguna pobreza", fontsize=11, color="#2C3E50", ha='center', va='center')

label_names = [
    "Fuera de pobreza",
    "Pobreza por ingresos",
    "Pobreza multidimensional",
    "Pobreza por ingresos y\nmultidimensional"
]

for i, (pct, val) in enumerate(zip(pcts_a, values_a)):
    angle_mid = 90 - sum(pcts_a[:i]) - pct / 2
    angle_deg = angle_mid % 360
    angle_rad = np.deg2rad(angle_deg)
    
    x = np.cos(angle_rad)
    y = np.sin(angle_rad)
    ha = 'left' if x > 0 else 'right'
    va = 'center'
    
    dist_text = 1.25
    x_text = dist_text * np.cos(angle_rad)
    y_text = dist_text * np.sin(angle_rad)
    
    if label_names[i] == "Pobreza por ingresos y\nmultidimensional":
        x_text = -0.5
        y_text = 1.25
        ha = 'right'
    elif label_names[i] == "Pobreza multidimensional":
        x_text = -1.1
        y_text = 0.8
        ha = 'right'
    elif label_names[i] == "Pobreza por ingresos":
        x_text = -1.25
        y_text = 0.0
        ha = 'right'
    elif label_names[i] == "Fuera de pobreza":
        x_text = 1.25
        y_text = 0.0
        ha = 'left'
        
    ax_a.annotate(
        f"{label_names[i]}\n{pct:.1f}%",
        xy=(x * 0.9, y * 0.9),
        xytext=(x_text, y_text),
        arrowprops=dict(arrowstyle="-", color="#BDC3C7", linewidth=0.8),
        fontsize=11.5,
        color="#1A2530",
        fontweight='bold',
        ha=ha,
        va=va
    )

ax_a.set_title("A. Distribución nacional de los hogares según condición de pobreza\n", fontsize=16, fontweight='bold', color="#1A2530", pad=20)
plt.figtext(0.5, 0.91, "% de hogares según cruce pobreza por ingresos × pobreza multidimensional (CASEN 2024).", fontsize=11, color="#2C3E50", ha='center')

n_text = f"n = {n_ponderado:,.0f} hogares (población expandida) - {n_muestral:,} hogares en la muestra"
plt.figtext(0.5, 0.05, n_text, fontsize=9.5, color="#7F8C8D", ha='center', style='italic')

ax_a.axis('equal')
plt.xlim(-1.7, 1.7)
plt.ylim(-1.5, 1.5)
ax_a.axis('off')

# plt.savefig(DIR_OUT / "grafico_A_donut.png", dpi=300, bbox_inches='tight')
# plt.savefig(DIR_OUT / "grafico_A_donut.pdf", bbox_inches='tight')
plt.close(fig_a)
print("-> Gráfico A (Donut) omitido según requerimientos de no regenerar otros gráficos.")

# %% 2. Gráfico 1: mapa de sobrerrepresentación
print("Generando Gráfico 1: Sobrerrepresentación territorial...")

# Definir colores planos por categoría
colores_flat = {
    "Pobreza por ingresos": "#F39D2B",
    "Pobreza multidimensional": "#E64F42",
    "Pobreza ingresos y multidim.": "#731819"
}

gdf_plot["color"] = gdf_plot["cat_sobrerep"].map(colores_flat)

fig_b = plt.figure(figsize=(18, 20))
fig_b.suptitle("C. Tipo de pobreza más sobrerrepresentado por región\n", fontsize=24, fontweight='bold', color=COLOR_TEXTO_FUERTE, y=0.98)
plt.figtext(0.5, 0.94, "Cada región se colorea según el tipo de pobreza cuya proporción supera en mayor medida el promedio nacional.", fontsize=18, color=COLOR_TEXTO, ha='center')

# Grilla asimétrica: 3 filas x 4 columnas
gs = GridSpec(3, 4, figure=fig_b, wspace=0.3, hspace=0.3, height_ratios=[1, 1.5, 1])

ax_ng = fig_b.add_subplot(gs[0, 0:2])
ax_nc = fig_b.add_subplot(gs[0, 2:4])
ax_zc = fig_b.add_subplot(gs[1, 1:3]) # Zona central más grande y centrada
ax_zs = fig_b.add_subplot(gs[2, 0:2])
ax_za = fig_b.add_subplot(gs[2, 2:4])

# Leyendas al lado de la Zona Central
ax_leg1 = fig_b.add_subplot(gs[1, 0])
ax_leg2 = fig_b.add_subplot(gs[1, 3])
ax_leg1.axis('off')
ax_leg2.axis('off')

# Establecer límites explícitos para dibujar con coordenadas normalizadas
ax_leg1.set_xlim(0, 1)
ax_leg1.set_ylim(0, 1)
ax_leg2.set_xlim(0, 1)
ax_leg2.set_ylim(0, 1)

zonas_axes = [
    ("Norte Grande", ax_ng),
    ("Norte Chico", ax_nc),
    ("Zona Central", ax_zc),
    ("Zona Sur", ax_zs),
    ("Zona Austral", ax_za)
]

for zona_name, ax in zonas_axes:
    sub_gdf = gdf_plot[gdf_plot["zona"] == zona_name]
    sub_gdf.plot(ax=ax, color=sub_gdf["color"], edgecolor='black', linewidth=0.8)
    
    if zona_name == "Zona Central":
        # Recortar islas remotas (Isla de Pascua y Juan Fernández) para igualar el tamaño visual del continente
        minx, miny, maxx, maxy = sub_gdf.total_bounds
        # El continente no pasa de -73.5 aprox, forzamos minx para excluir las islas
        minx_continental = max(minx, -73.5)
        x_margin = (maxx - minx_continental) * 0.05
        y_margin = (maxy - miny) * 0.05
        ax.set_xlim(minx_continental - x_margin, maxx + x_margin)
        ax.set_ylim(miny - y_margin, maxy + y_margin)
        
    ax.set_title(zona_name, fontsize=18, fontweight='bold', color=COLOR_TEXTO_FUERTE)
    ax.axis('off')

# Dibujar la leyenda simple en ax_leg1
ax_leg1.text(0.0, 0.95, "Tipo de pobreza más\nsobrerrepresentado", fontsize=14, fontweight='bold', color=COLOR_TEXTO_FUERTE, va='top', ha='left')

box_h = 0.04
box_w = 0.08
y_top = 0.80

map_legend_labels = {
    "Pobreza por ingresos": "Pobreza por ingresos",
    "Pobreza multidimensional": "Pobreza multidimensional",
    "Pobreza ingresos y multidim.": "Pobreza por ingresos y multidimensional"
}

for i, (cat, color) in enumerate(colores_flat.items()):
    y_pos = y_top - i * 0.12
    rect = plt.Rectangle((0.0, y_pos - box_h), box_w, box_h, facecolor=color, edgecolor='black', linewidth=0.5)
    ax_leg1.add_patch(rect)
    display_cat = map_legend_labels.get(cat, cat)
    ax_leg1.text(box_w + 0.04, y_pos - box_h/2, display_cat, fontsize=11, fontweight='bold', color=COLOR_TEXTO, va='center', ha='left')

# Dibujar la tabla de tamaños de muestra en ax_leg2
ax_leg2.text(0.0, 0.97, "Magnitud de Muestra por Región", fontsize=14, fontweight='bold', color=COLOR_TEXTO_FUERTE, va='top')
ax_leg2.text(0.0, 0.93, "Hogares pobres en la muestra (n) y expandidos (N)", fontsize=10.5, color=COLOR_TEXTO, va='top')

# Encabezados
ax_leg2.text(0.0, 0.88, "Región", fontsize=9.5, fontweight='bold', color=COLOR_TEXTO_FUERTE)
ax_leg2.text(0.60, 0.88, "Muestra (n)", fontsize=9.5, fontweight='bold', color=COLOR_TEXTO_FUERTE, ha='right')
ax_leg2.text(0.98, 0.88, "Expandida (N)", fontsize=9.5, fontweight='bold', color=COLOR_TEXTO_FUERTE, ha='right')

# Separador
ax_leg2.plot([0, 0.98], [0.86, 0.86], color="#BDC3C7", linewidth=1)

y_pos = 0.82
zonas_orden = ["Norte Grande", "Norte Chico", "Zona Central", "Zona Sur", "Zona Austral"]
for zona_name in zonas_orden:
    ax_leg2.text(0.0, y_pos, f"■ {zona_name}", fontsize=9.5, fontweight='bold', color="#2C3E50")
    y_pos -= 0.022
    
    sub_df = gdf_plot[gdf_plot["zona"] == zona_name].drop_duplicates(subset=["region"]).sort_values("region")
    for _, row in sub_df.iterrows():
        cod = MAPA_SIGLAS_REGIONALES.get(row["region"], str(row["region"]))
        name = row["region_name"]
        n_m = row["n_sample_poor"]
        n_p = row["n_expanded_poor"]
        
        ax_leg2.text(0.05, y_pos, f"{cod}: {name}", fontsize=8.5, color=COLOR_TEXTO)
        ax_leg2.text(0.60, y_pos, f"{n_m:,}", fontsize=8.5, color=COLOR_TEXTO, ha='right')
        ax_leg2.text(0.98, y_pos, f"{n_p:,.0f}", fontsize=8.5, color=COLOR_TEXTO, ha='right')
        y_pos -= 0.022
    y_pos -= 0.015

output_path_c = DIR_OUT / "grafico_C_tipo_pobreza_mas_sobrerrepresentado.png"
plt.savefig(output_path_c, dpi=300, bbox_inches='tight')
plt.savefig(DIR_OUT / "grafico_C_tipo_pobreza_mas_sobrerrepresentado.pdf", bbox_inches='tight')
# También exportar al nombre corregido anterior para no romper flujos que dependan de él
plt.savefig(DIR_OUT / "grafico_C_sobrerrepresentacion_corregido.png", dpi=300, bbox_inches='tight')
plt.savefig(DIR_OUT / "grafico_C_sobrerrepresentacion_corregido.pdf", bbox_inches='tight')
plt.close(fig_b)
print(f"-> Exportado: {output_path_c}")
print("-> Exportado: grafico_C_sobrerrepresentacion_corregido.png")

# %% 3. Gráfico 2: dispersión de Moran
print("Generando Gráfico 2: Dispersión de Moran...")

fig_m, ax_m = plt.figure(figsize=(10, 8)), plt.gca()

quad_colors = {
    "Alto-Alto": "#E74C3C",  
    "Bajo-Bajo": "#2E86C1",  
    "Alto-Bajo": "#F5B7B1",  
    "Bajo-Alto": "#85C1E9"   
}

ax_m.axvline(0, color="#BDC3C7", linestyle="-", linewidth=1.2)
ax_m.axhline(0, color="#BDC3C7", linestyle="-", linewidth=1.2)

slope, intercept = np.polyfit(gdf_spatial["y_std"], gdf_spatial["spatial_lag"], 1)
x_line = np.array([gdf_spatial["y_std"].min() - 0.5, gdf_spatial["y_std"].max() + 0.5])
y_line = slope * x_line + intercept
ax_m.plot(x_line, y_line, color="#7F8C8D", linestyle="--", linewidth=1.5, zorder=1)

OFFSETS_MORAN = {
    "VIII": (-0.05, 0.08, "center", "bottom"),  # Biobío
    "IX": (-0.12, -0.05, "right", "center"),    # La Araucanía
    "XIV": (-0.12, 0.05, "right", "center"),    # Los Ríos
    "XVI": (0.12, -0.05, "left", "center"),     # Ñuble
    "RM": (0.12, -0.05, "left", "center"),      # Metropolitana
    "VI": (-0.12, 0.05, "right", "center"),     # O'Higgins
    "III": (0.12, 0.05, "left", "center")       # Atacama
}

for _, row in gdf_spatial.iterrows():
    if row["y_std"] > 0 and row["spatial_lag"] > 0: q = "Alto-Alto"
    elif row["y_std"] < 0 and row["spatial_lag"] < 0: q = "Bajo-Bajo"
    elif row["y_std"] > 0 and row["spatial_lag"] < 0: q = "Alto-Bajo"
    elif row["y_std"] < 0 and row["spatial_lag"] > 0: q = "Bajo-Alto"
    else: q = "No significativo"
    
    color = quad_colors.get(q, "#95A5A6")
    ax_m.scatter(row["y_std"], row["spatial_lag"], s=100, color=color, edgecolor='white', linewidth=1.0, zorder=3)
    
    cod = row["cod_region"]
    dx, dy, ha, va = 0.0, 0.08, "center", "bottom"
    if cod in OFFSETS_MORAN:
        dx, dy, ha, va = OFFSETS_MORAN[cod]
        
    ax_m.text(row["y_std"] + dx, row["spatial_lag"] + dy, cod, 
              fontsize=9, color=COLOR_TEXTO, zorder=4, ha=ha, va=va, fontweight='bold')

# Etiquetas de cuadrantes
ax_m.text(x_line.max()*0.8, y_line.max()*0.9, "Alto-Alto", fontsize=14, color=quad_colors["Alto-Alto"], fontweight='bold', alpha=0.8, ha='center', va='center')
ax_m.text(x_line.min()*0.8, y_line.max()*0.9, "Bajo-Alto", fontsize=14, color=quad_colors["Bajo-Alto"], fontweight='bold', alpha=0.8, ha='center', va='center')
ax_m.text(x_line.min()*0.8, y_line.min()*0.9, "Bajo-Bajo", fontsize=14, color=quad_colors["Bajo-Bajo"], fontweight='bold', alpha=0.8, ha='center', va='center')
ax_m.text(x_line.max()*0.8, y_line.min()*0.9, "Alto-Bajo", fontsize=14, color=quad_colors["Alto-Bajo"], fontweight='bold', alpha=0.8, ha='center', va='center')

stats_text = f"Moran's I: {moran.I:.3f}\np-value: {moran.p_sim:.4f}"
ax_m.text(0.03, 0.95, stats_text, transform=ax_m.transAxes, fontsize=12, 
          verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#BDC3C7", linewidth=1))

ax_m.set_title("Autocorrelación espacial global de la pobreza multidimensional\n", fontsize=16, fontweight='bold', color=COLOR_TEXTO_FUERTE, loc='left')
plt.figtext(0.125, 0.89, "Dispersión de Moran por región (CASEN 2024)", fontsize=12, color=COLOR_TEXTO)

ax_m.set_xlabel("Pobreza multidimensional regional estandarizada (Z-score)", fontsize=12, labelpad=10, color=COLOR_TEXTO_FUERTE)
ax_m.set_ylabel("Rezago espacial de la pobreza", fontsize=12, labelpad=10, color=COLOR_TEXTO_FUERTE)

ax_m.spines['top'].set_visible(False)
ax_m.spines['right'].set_visible(False)
ax_m.grid(True, linestyle=':', color="#E5E7E9", zorder=0)

output_path_moran = DIR_OUT / "grafico_moran_dispersion_mejorado.png"
# plt.savefig(output_path_moran, dpi=300, bbox_inches='tight')
# plt.savefig(DIR_OUT / "grafico_moran_dispersion_mejorado.pdf", bbox_inches='tight')
plt.close(fig_m)
print("-> Gráfico Moran omitido según requerimientos de no regenerar otros gráficos.")

# %% 4. Gráfico 3: mapa LISA mejorado
print("Generando Gráfico 3: Mapa LISA...")

lisa_colors_map = {
    "Alto-Alto": "#E74C3C",       # Rojo
    "Bajo-Bajo": "#2E86C1",       # Azul
    "Alto-Bajo": "#F5B7B1",       # Rosado claro
    "Bajo-Alto": "#85C1E9",       # Celeste
    "No significativo": "#E5E7E9" # Gris claro
}

lisa_labels_map = {
    "Alto-Alto": "Alto-Alto: alta pobreza rodeada de alta pobreza",
    "Bajo-Bajo": "Bajo-Bajo: baja pobreza rodeada de baja pobreza",
    "Alto-Bajo": "Alto-Bajo: alta pobreza rodeada de baja pobreza",
    "Bajo-Alto": "Bajo-Alto: baja pobreza rodeada de alta pobreza",
    "No significativo": "No significativo"
}

fig_l, ax_l = plt.subplots(figsize=(10, 16))

ax_l.set_title("Clústeres espaciales de pobreza multidimensional\n", fontsize=20, fontweight='bold', color=COLOR_TEXTO_FUERTE, loc='center')
plt.figtext(0.5, 0.87, "Mapa LISA por región (CASEN 2024)", fontsize=14, color=COLOR_TEXTO, ha='center')

gdf_spatial.plot(ax=ax_l, color=gdf_spatial["lisa_class"].map(lisa_colors_map), edgecolor='white', linewidth=1.0)
ax_l.axis('off')

# Zoom para aprovechar el espacio (ajustar márgenes)
bounds = gdf_spatial.total_bounds
x_marg = (bounds[2] - bounds[0]) * 0.02
y_marg = (bounds[3] - bounds[1]) * 0.02
ax_l.set_xlim(bounds[0] - x_marg, bounds[2] + x_marg)
ax_l.set_ylim(bounds[1] - y_marg, bounds[3] + y_marg)

# Etiquetar solo significativas
for _, row in gdf_spatial[gdf_spatial["lisa_class"] != "No significativo"].iterrows():
    rep = row["geometry"].representative_point()
    cod = row["cod_region"]
    if cod == "XV":
        # Línea de llamada (callout) y texto a la derecha para XV
        ax_l.annotate(
            cod, 
            xy=(rep.x, rep.y), 
            xytext=(rep.x + 1.5, rep.y),
            arrowprops=dict(arrowstyle="-", color=COLOR_TEXTO_FUERTE, linewidth=0.8),
            fontsize=10, fontweight='bold', color=COLOR_TEXTO_FUERTE,
            va='center', ha='left'
        )
    else:
        # Texto con efecto de halo blanco para las demás regiones significativas
        ax_l.text(
            rep.x, rep.y, cod, 
            fontsize=10, fontweight='bold', color=COLOR_TEXTO_FUERTE, 
            ha='center', va='center',
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")]
        )

# Leyenda dinámica
categorias_presentes = gdf_spatial["lisa_class"].unique()
leyenda_lisa = []

# Forzar el orden lógico si están presentes
for cat in ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo", "Bajo-Alto", "No significativo"]:
    if cat in categorias_presentes:
        leyenda_lisa.append(Patch(facecolor=lisa_colors_map[cat], edgecolor='black', label=lisa_labels_map[cat]))

ax_l.legend(handles=leyenda_lisa, loc='lower left', fontsize=12, frameon=True, facecolor='white', edgecolor="#BDC3C7", borderpad=1, title="Clasificación LISA", title_fontsize=13)

output_path_lisa = DIR_OUT / "grafico_lisa_mejorado.png"
# plt.savefig(output_path_lisa, dpi=300, bbox_inches='tight')
# plt.savefig(DIR_OUT / "grafico_lisa_mejorado.pdf", bbox_inches='tight')
plt.close(fig_l)
print("-> Gráfico LISA omitido según requerimientos de no regenerar otros gráficos.")

# %% 5. Gráfico F: Radar Norte Grande
print("Generando Gráfico F: Radar Norte Grande (Radar)...")
import plotly.graph_objects as go

# Cargar g4_radar
df_radar = pd.read_csv(DIR_PROCESSED / "g4_radar.csv")

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

# Formato N muestral
n_mues = 9373
n_pond = sum(r["n_pond"] for r in resultados.values())
def _formato_n_radar(n_muestral: int, n_ponderado: float, unidad: str = "hogares") -> str:
    return (f"n = {n_ponderado:,.0f} {unidad} (población expandida) · "
            f"{n_muestral:,} {unidad} en la muestra")
n_label = _formato_n_radar(n_mues, n_pond)

# Colores de la entrega anterior
COLOR_URBANO = "#175884"     # azul
COLOR_RURAL = "#D3541F"      # naranja
FONT_FAMILY = "Inter, Roboto, sans-serif"

TEXTO_OSCURO = "#1A1A1A"
TEXTO_GRIS = "#4A4A4A"

fig_radar = go.Figure()

# Traza Hogares Chilenos (azul)
fig_radar.add_trace(go.Scatterpolar(
    r=chil_v, theta=ejes, mode="lines+markers",
    name=f"Hogares Chilenos (jefe nacido en Chile) (n={resultados['Hogares chilenos']['n_hog']:,})",
    line=dict(color=COLOR_URBANO, width=2.4),
    marker=dict(size=6, color=COLOR_URBANO, symbol="circle",
                line=dict(color="white", width=0.8)),
    fill="toself", fillcolor="rgba(43,91,132,0.22)",
    hovertemplate="<b>%{theta}</b><br>Hogares Chilenos: %{r:.3f}<extra></extra>",
))

# Traza Hogares Inmigrantes (naranja)
fig_radar.add_trace(go.Scatterpolar(
    r=inmi_v, theta=ejes, mode="lines+markers",
    name=f"Hogares Inmigrantes (jefe nacido en el extranjero) (n={resultados['Hogares inmigrantes']['n_hog']:,})",
    line=dict(color=COLOR_RURAL, width=2.4),
    marker=dict(size=6, color=COLOR_RURAL, symbol="square",
                line=dict(color="white", width=0.8)),
    fill="toself", fillcolor="rgba(211,84,0,0.32)",
    hovertemplate="<b>%{theta}</b><br>Hogares Inmigrantes: %{r:.3f}<extra></extra>",
))

def _titulo_a4_radar(titulo: str, subtitulo: str, size_t: int = 14, size_s: int = 9.5):
    return dict(
        text=f"<b>{titulo}</b><br><sup>{subtitulo}</sup>",
        font=dict(family=FONT_FAMILY, size=size_t, color=TEXTO_OSCURO),
        x=0.02, xanchor="left", y=0.97, yanchor="top",
        pad=dict(t=5, b=5),
    )

fig_radar.update_layout(
    polar=dict(
        bgcolor="#FAFAFA",
        radialaxis=dict(
            visible=True, range=[0, escala_max],
            tickvals=list(tickvals),
            ticktext=ticktext,
            tickfont=dict(size=7.5, color=TEXTO_OSCURO, family=FONT_FAMILY),
            angle=30, tickangle=0,
            gridcolor="#E5E7E9", linecolor="#E5E7E9"
        ),
        angularaxis=dict(
            tickfont=dict(size=8.5, color=TEXTO_OSCURO, family=FONT_FAMILY),
            rotation=90, direction="clockwise",
            linecolor="#BDC3C7", gridcolor="#E5E7E9"
        ),
    ),
    title=_titulo_a4_radar(
        "F. Intensidad de carencias por dimensión en Norte Grande",
        "Promedio ponderado del número de carencias de la metodología multidimensional 2015."
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=-0.12,
        xanchor="center", x=0.5,
        font=dict(size=8.5, family=FONT_FAMILY, color=TEXTO_OSCURO),
        title=dict(text="Tipos de hogar", side="top", font=dict(size=8, color=TEXTO_OSCURO))
    ),
    font=dict(family=FONT_FAMILY, color=TEXTO_OSCURO),
    margin=dict(t=60, b=120, l=50, r=50),
    height=506, width=620,  # A4 W, H
    paper_bgcolor="white", plot_bgcolor="white",
    annotations=[
        dict(x=0.5, y=-0.20, xref="paper", yref="paper",
             text="Unidad: carencias promedio por hogar.", showarrow=False,
             xanchor="center",
             font=dict(size=8.0, color=TEXTO_GRIS, family=FONT_FAMILY)),
        dict(x=0.5, y=-0.29, xref="paper", yref="paper",
             text=f"<i>{n_label}</i>", showarrow=False,
             xanchor="center",
             font=dict(size=7.5, color="#7F8C8D", family=FONT_FAMILY))
    ],
)

output_radar = DIR_OUT / "Radar.png"
# fig_radar.write_image(str(output_radar), width=620, height=506, scale=2.5)
print("-> Gráfico Radar omitido según requerimientos de no regenerar otros gráficos.")

print("¡Proceso completado exitosamente!")
