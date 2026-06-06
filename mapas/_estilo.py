# =============================================================================
#  CASEN 2024 · Sistema de mapas (Biobio + Nuble) - Estilo compartido
#  Paleta y rutas reutilizadas del resto del entregable para coherencia visual.
# =============================================================================
from pathlib import Path

# --- Rutas ------------------------------------------------------------------
DIR_MAPAS = Path(__file__).resolve().parent
DIR_RAIZ = DIR_MAPAS.parent
DIR_GEO = DIR_MAPAS / "geo"
DIR_DATOS = DIR_MAPAS / "datos"
DIR_FINAL = DIR_RAIZ / "Final"

CASEN_PRINCIPAL = DIR_RAIZ / "casen_2024.dta"
CASEN_PROV_COMUNA = DIR_RAIZ / "casen_2024_provincia_comuna.dta"
GEOJSON_PROVINCIAS = DIR_GEO / "provincias_biobio_nuble.geojson"
CSV_INDICADORES = DIR_DATOS / "indicadores_provincia.csv"

for _d in (DIR_GEO, DIR_DATOS, DIR_FINAL):
    _d.mkdir(parents=True, exist_ok=True)

# --- Identidad visual (consistente con generador_final_4graficos.py) --------
FONT_FAMILY = "Inter, Roboto, sans-serif"
COLOR_TEXTO = "#2C3E50"
COLOR_TEXTO_FUERTE = "#1A2530"
COLOR_URBANO = "#2B5B84"   # azul corporativo
COLOR_RURAL = "#D35400"    # naranja
COLOR_MULTI = "#E74C3C"    # rojo institucional del proyecto
COLOR_NEUTRO = "#BDC3C7"

# Escala secuencial roja (misma del heatmap regional del proyecto)
ESCALA_SECUENCIAL = ["#FDEDEC", "#F5B7B1", "#E74C3C", "#78281F"]
# Escala divergente azul-blanco-rojo (centrada en 0)
ESCALA_DIVERGENTE = [COLOR_URBANO, "#7FB3D5", "#F4F6F7", "#E59866", COLOR_MULTI]

FUENTE_NOTA = "Fuente: Encuesta CASEN 2024 (MDSF) - elaboracion propia."

# --- Provincias de las regiones de Biobio (8) y Nuble (16) ------------------
# Codigos de 3 digitos de la base casen_2024_provincia_comuna.dta
PROVINCIAS = {
    81:  {"nombre": "Concepcion", "region": "Biobio"},
    82:  {"nombre": "Arauco",     "region": "Biobio"},
    83:  {"nombre": "Biobio",     "region": "Biobio"},
    161: {"nombre": "Diguillin",  "region": "Nuble"},
    162: {"nombre": "Punilla",    "region": "Nuble"},
    163: {"nombre": "Itata",      "region": "Nuble"},
}
CODIGOS_REGION = [8, 16]
CRS_METRICO = "EPSG:32718"  # UTM 18S, apropiado para Chile centro-sur


def aplicar_estilo_matplotlib():
    """Aplica una configuracion base de matplotlib coherente con el entregable."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "Roboto", "DejaVu Sans"],
        "text.color": COLOR_TEXTO,
        "axes.edgecolor": "#DFE6E9",
        "axes.labelcolor": COLOR_TEXTO,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
