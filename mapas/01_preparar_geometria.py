# =============================================================================
#  FASE 1 · Preparacion de la geometria administrativa
#  Descarga limites comunales de Chile (division politica vigente, 16 regiones),
#  recorta a las regiones de Biobio (8) y Nuble (16) y los disuelve a nivel
#  PROVINCIA (6 unidades).
#  Salida: mapas/geo/provincias_biobio_nuble.geojson
# =============================================================================
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

# Fuente: pachamaltese/chilemaps. GeoJSON de comunas con la division politica
# vigente (incluye la Region de Nuble) y codigos administrativos estandar INE:
# codigo_comuna (5 digitos), codigo_provincia (3), codigo_region (2).
URL_BASE = ("https://raw.githubusercontent.com/pachamaltese/chilemaps/"
            "master/data_geojson/comunas")
DIR_FUENTE = st.DIR_GEO / "fuente"
# Region 8 = r08, Region 16 = r16
ARCHIVOS_REGION = {8: "r08.geojson", 16: "r16.geojson"}


def descargar_fuente():
    """Descarga los GeoJSON de comunas de las regiones 8 y 16 (idempotente)."""
    DIR_FUENTE.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for cod_region, archivo in ARCHIVOS_REGION.items():
        destino = DIR_FUENTE / f"chilemaps_{archivo}"
        rutas[cod_region] = destino
        if destino.exists():
            print(f"  Fuente ya presente: {destino.name}")
            continue
        urllib.request.urlretrieve(f"{URL_BASE}/{archivo}", destino)
        print(f"  Descargado: {destino.name} ({destino.stat().st_size:,} bytes)")
    return rutas


def preparar_provincias():
    print("FASE 1 | Preparando geometria de provincias (Biobio + Nuble)")
    rutas = descargar_fuente()

    # Cargar y unir las comunas de ambas regiones
    capas = [gpd.read_file(r) for r in rutas.values()]
    comunas = gpd.GeoDataFrame(pd.concat(capas, ignore_index=True), crs=capas[0].crs)
    comunas["cod_provincia"] = comunas["codigo_provincia"].astype(int)
    comunas["cod_region"] = comunas["codigo_region"].astype(int)
    print(f"  Comunas cargadas: {len(comunas)} "
          f"(regiones {sorted(comunas['cod_region'].unique())})")

    # Disolver a nivel provincia
    provincias = comunas.dissolve(by="cod_provincia", as_index=False)
    provincias = provincias[["cod_provincia", "cod_region", "geometry"]]

    # Nombres legibles desde el modulo de estilo
    provincias["provincia"] = provincias["cod_provincia"].map(
        lambda c: st.PROVINCIAS[c]["nombre"])
    provincias["region"] = provincias["cod_provincia"].map(
        lambda c: st.PROVINCIAS[c]["region"])

    # Limpieza topologica menor por si el dissolve deja geometrias invalidas
    provincias["geometry"] = provincias["geometry"].buffer(0)
    provincias = provincias.set_crs("EPSG:4326", allow_override=True)

    # Validaciones
    assert len(provincias) == 6, f"Se esperaban 6 provincias, hay {len(provincias)}"
    assert set(provincias["cod_provincia"]) == set(st.PROVINCIAS), \
        "Los codigos de provincia no coinciden con los esperados"
    assert provincias.is_valid.all(), "Hay geometrias invalidas tras el dissolve"

    provincias = provincias.sort_values("cod_provincia").reset_index(drop=True)
    provincias.to_file(st.GEOJSON_PROVINCIAS, driver="GeoJSON")

    print(f"  Guardado: {st.GEOJSON_PROVINCIAS}")
    print(provincias[["cod_provincia", "provincia", "region"]].to_string(index=False))
    return provincias


if __name__ == "__main__":
    preparar_provincias()
    print("FASE 1 completada.\n")
