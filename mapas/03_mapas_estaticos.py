# =============================================================================
#  FASE 3 · Mapas tematicos estaticos (GeoPandas + Matplotlib)
#  Genera 4 mapas coropleticos de las 6 provincias de Biobio y Nuble, en calidad
#  de impresion (300 dpi), a partir de la geometria (Fase 1) y los indicadores
#  CASEN 2024 (Fase 2).
#  Salida: Final/mapa_1_pobreza_multi.png ... mapa_4_salud.png
# =============================================================================
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.patches import Circle

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

st.aplicar_estilo_matplotlib()

CMAP_SECUENCIAL = LinearSegmentedColormap.from_list("rojo_proy", st.ESCALA_SECUENCIAL)
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list("div_proy", st.ESCALA_DIVERGENTE)
CMAP_SALUD = LinearSegmentedColormap.from_list(
    "azul_proy", ["#EAF1F6", "#9DC3D9", st.COLOR_URBANO, "#16314A"])

FIGSIZE = (8.6, 10.0)
MAP_RECT = [0.035, 0.055, 0.93, 0.80]   # rectangulo del eje del mapa
BBOX_TXT = dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78,
                edgecolor="none")


# -----------------------------------------------------------------------------
# Utilidades comunes
# -----------------------------------------------------------------------------
def cargar_datos():
    """Une la geometria de provincias con los indicadores y proyecta a metros."""
    geo = gpd.read_file(st.GEOJSON_PROVINCIAS)
    ind = pd.read_csv(st.CSV_INDICADORES)
    gdf = geo.merge(ind, on=["cod_provincia", "provincia", "region"])
    return gdf.to_crs(st.CRS_METRICO)


def nueva_figura(titulo, subtitulo):
    """Figura con titulo, subtitulo, nota de fuente y eje de mapa posicionado."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(MAP_RECT)
    ax.set_axis_off()
    fig.text(0.035, 0.95, titulo, fontsize=16, fontweight="bold",
             color=st.COLOR_TEXTO_FUERTE, ha="left")
    fig.text(0.035, 0.905, subtitulo, fontsize=10.5, color=st.COLOR_TEXTO,
             ha="left", va="top")
    fig.text(0.035, 0.02, st.FUENTE_NOTA, fontsize=8.5, color="#7F8C8D",
             style="italic", ha="left")
    return fig, ax


def etiquetar(ax, gdf, texto_func):
    """Escribe el nombre de cada provincia + un valor en su punto interior.
    El texto va siempre en tono oscuro sobre un recuadro blanco semitransparente,
    de modo que es legible sobre cualquier color de relleno."""
    for _, fila in gdf.iterrows():
        punto = fila.geometry.representative_point()
        ax.annotate(texto_func(fila), xy=(punto.x, punto.y),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color=st.COLOR_TEXTO_FUERTE, linespacing=1.3,
                    bbox=BBOX_TXT, zorder=6)


def barra_color(ax, cmap, norm, etiqueta):
    """Colorbar vertical compacta incrustada sobre el oceano (esquina sup. izq.)."""
    cax = ax.inset_axes([0.045, 0.55, 0.035, 0.36])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = ax.figure.colorbar(sm, cax=cax)
    cbar.set_label(etiqueta, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor("#BDC3C7")
    return cbar


def guardar(fig, nombre):
    ruta = st.DIR_FINAL / nombre
    fig.savefig(ruta, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  Guardado: {ruta.name}")


# -----------------------------------------------------------------------------
# MAPA 1 · Coropletico secuencial — Pobreza multidimensional
# -----------------------------------------------------------------------------
def mapa_1(gdf):
    col = "pobreza_multi_pct"
    norm = Normalize(vmin=gdf[col].min() * 0.95, vmax=gdf[col].max() * 1.02)
    fig, ax = nueva_figura(
        "Pobreza multidimensional en Biobio y Nuble",
        "Porcentaje de la poblacion en hogares con pobreza multidimensional, "
        "por provincia (CASEN 2024)")

    gdf.plot(ax=ax, column=col, cmap=CMAP_SECUENCIAL, norm=norm,
             edgecolor="white", linewidth=1.4)
    etiquetar(ax, gdf, lambda f: f"{f['provincia']}\n{f[col]:.1f}%")
    barra_color(ax, CMAP_SECUENCIAL, norm,
                "% poblacion en pobreza multidimensional")
    guardar(fig, "mapa_1_pobreza_multi.png")


# -----------------------------------------------------------------------------
# MAPA 2 · Coropletico divergente — Brecha de dependencia de subsidios
# -----------------------------------------------------------------------------
def mapa_2(gdf):
    col = "dep_subsidios_brecha"
    lim = max(abs(gdf[col].min()), abs(gdf[col].max())) * 1.05
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    media_zona = gdf["dep_subsidios_pct"].iloc[0] - gdf[col].iloc[0]
    fig, ax = nueva_figura(
        "Dependencia de subsidios estatales: brecha territorial",
        f"Desviacion de cada provincia respecto al promedio de la macrozona "
        f"({media_zona:.1f}% del ingreso del hogar). CASEN 2024")

    gdf.plot(ax=ax, column=col, cmap=CMAP_DIVERGENTE, norm=norm,
             edgecolor="white", linewidth=1.4)

    def _txt(f):
        signo = "+" if f[col] >= 0 else ""
        return (f"{f['provincia']}\n{f['dep_subsidios_pct']:.1f}%\n"
                f"({signo}{f[col]:.1f} pp)")

    etiquetar(ax, gdf, _txt)
    barra_color(ax, CMAP_DIVERGENTE, norm,
                "Brecha vs. promedio macrozona (puntos porcentuales)")
    ax.annotate("Azul: dependencia bajo el promedio\n"
                "Rojo: dependencia sobre el promedio",
                xy=(0.045, 0.50), xycoords="axes fraction", fontsize=8.5,
                color=st.COLOR_TEXTO, ha="left", va="top")
    guardar(fig, "mapa_2_dependencia_subsidios.png")


# -----------------------------------------------------------------------------
# MAPA 3 · Mapa bivariado — Ingreso autonomo p/c  x  Dependencia de subsidios
# -----------------------------------------------------------------------------
def _paleta_bivariada():
    """Grilla 3x3 por interpolacion bilineal entre 4 esquinas.
    Filas (i) = ingreso autonomo: 0 bajo -> 2 alto.
    Columnas (j) = dependencia de subsidios: 0 baja -> 2 alta.
    """
    c00 = np.array(to_rgb("#F3E6D6"))  # ingreso bajo  + dependencia baja
    c02 = np.array(to_rgb("#78281F"))  # ingreso bajo  + dependencia alta
    c20 = np.array(to_rgb("#D7E3EC"))  # ingreso alto  + dependencia baja
    c22 = np.array(to_rgb("#6C5B7B"))  # ingreso alto  + dependencia alta
    grilla = {}
    for i in range(3):
        ti = i / 2
        for j in range(3):
            tj = j / 2
            top = c00 * (1 - tj) + c02 * tj
            bot = c20 * (1 - tj) + c22 * tj
            grilla[(i, j)] = tuple(top * (1 - ti) + bot * ti)
    return grilla


def _terciles(serie):
    q1, q2 = serie.quantile([1 / 3, 2 / 3])
    return serie.apply(lambda v: 0 if v <= q1 else (1 if v <= q2 else 2))


def mapa_3(gdf):
    gdf = gdf.copy()
    gdf["ti_ingreso"] = _terciles(gdf["ing_autonomo_pc"])
    gdf["tj_dep"] = _terciles(gdf["dep_subsidios_pct"])
    grilla = _paleta_bivariada()
    gdf["color_bivar"] = gdf.apply(
        lambda f: grilla[(int(f["ti_ingreso"]), int(f["tj_dep"]))], axis=1)

    fig, ax = nueva_figura(
        "Doble vulnerabilidad: ingreso propio vs. dependencia estatal",
        "Cruce del ingreso autonomo per capita del hogar y el peso de los "
        "subsidios en el ingreso, por provincia (CASEN 2024)")
    gdf.plot(ax=ax, color=gdf["color_bivar"].tolist(),
             edgecolor="white", linewidth=1.4)
    etiquetar(ax, gdf,
              lambda f: (f"{f['provincia']}\n${f['ing_autonomo_pc']:,.0f}\n"
                         f"{f['dep_subsidios_pct']:.1f}% subsidios"))

    # Leyenda bivariada 3x3 incrustada
    lg = ax.inset_axes([0.62, 0.04, 0.22, 0.22])
    for i in range(3):
        for j in range(3):
            lg.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=grilla[(i, j)],
                                       edgecolor="white", linewidth=1.2))
    lg.set_xlim(0, 3)
    lg.set_ylim(0, 3)
    lg.set_xticks([])
    lg.set_yticks([])
    for spine in lg.spines.values():
        spine.set_visible(False)
    lg.set_xlabel("Dependencia de\nsubsidios  >>", fontsize=8,
                  color=st.COLOR_TEXTO)
    lg.set_ylabel("Ingreso autonomo\nper capita  >>", fontsize=8,
                  color=st.COLOR_TEXTO)
    lg.annotate("Doble vulnerabilidad\n(ingreso bajo +\ndependencia alta)",
                xy=(2.5, 0.5), xytext=(3.6, -1.6), fontsize=7.5,
                color="#78281F", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#78281F", lw=1.2))
    guardar(fig, "mapa_3_bivariado_ingreso_subsidios.png")


# -----------------------------------------------------------------------------
# MAPA 4 · Coropletico + simbolos proporcionales — Carencia de acceso a salud
# -----------------------------------------------------------------------------
def mapa_4(gdf):
    col = "salud_carencia_pct"
    norm = Normalize(vmin=gdf[col].min() * 0.92, vmax=gdf[col].max() * 1.05)
    fig, ax = nueva_figura(
        "Carencia de acceso a salud: tasa frente a volumen",
        "Color: % de poblacion en hogares con carencia de acceso a salud.\n"
        "Circulo: numero absoluto de personas afectadas. CASEN 2024")

    gdf.plot(ax=ax, column=col, cmap=CMAP_SALUD, norm=norm,
             edgecolor="#85929E", linewidth=0.8)

    # Simbolos proporcionales: el AREA del circulo es ~ poblacion afectada
    escala = 1500 / gdf["salud_carencia_pob"].max()  # s (puntos^2) por persona
    cent = gdf.geometry.representative_point()
    ax.scatter(cent.x, cent.y, s=gdf["salud_carencia_pob"] * escala,
               facecolor="white", alpha=0.35, edgecolor=st.COLOR_MULTI,
               linewidth=2.2, zorder=4)
    etiquetar(ax, gdf, lambda f: f"{f['provincia']}\n{f[col]:.1f}%")
    barra_color(ax, CMAP_SALUD, norm,
                "% poblacion con carencia de acceso a salud")

    # Leyenda de tamanos anidada. El eje de la leyenda usa coordenadas en
    # PUNTOS (mismo tamano fisico que el recuadro) para que los radios
    # coincidan exactamente con los circulos del mapa: r_pts = sqrt(s / pi).
    refs = [10000, 40000, 75000]
    lw_in, lh_in = 0.235 * FIGSIZE[0], 0.17 * FIGSIZE[1]
    lax = ax.inset_axes([0.018, 0.03, 0.235, 0.17])
    lax.set_xlim(0, lw_in * 72)
    lax.set_ylim(0, lh_in * 72)
    lax.set_xticks([])
    lax.set_yticks([])
    lax.set_facecolor("white")
    lax.patch.set_alpha(0.82)
    for spine in lax.spines.values():
        spine.set_edgecolor("#D5DBDB")
    lax.text(lw_in * 36, lh_in * 72 - 11, "Personas con carencia\nde acceso a salud",
             fontsize=8, ha="center", va="top", fontweight="bold",
             color=st.COLOR_TEXTO, linespacing=1.25)

    r_max = np.sqrt(max(refs) * escala / np.pi)
    cx, base = 8 + r_max, 9.0
    for ref in sorted(refs, reverse=True):  # mayor al fondo
        r = np.sqrt(ref * escala / np.pi)
        lax.add_patch(Circle((cx, base + r), r, facecolor="white", alpha=0.35,
                             edgecolor=st.COLOR_MULTI, linewidth=1.8))
        lax.plot([cx, cx + r_max + 8], [base + 2 * r, base + 2 * r],
                 color="#AEB6BF", linewidth=0.7)
        lax.text(cx + r_max + 11, base + 2 * r, f"{ref:,}", fontsize=7.5,
                 va="center", color=st.COLOR_TEXTO)
    guardar(fig, "mapa_4_salud.png")


def main():
    print("FASE 3 | Generando mapas tematicos estaticos")
    gdf = cargar_datos()
    mapa_1(gdf)
    mapa_2(gdf)
    mapa_3(gdf)
    mapa_4(gdf)
    print("FASE 3 completada.\n")


if __name__ == "__main__":
    main()
