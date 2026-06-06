# =============================================================================
#  FASE 2 · ETL de indicadores CASEN 2024 por provincia
#  Cruza la base CASEN con el archivo de provincia/comuna, recorta a Biobio (8)
#  y Nuble (16), y calcula indicadores ponderados por el factor de expansion
#  provincial (expp) a nivel PROVINCIA.
#  Salida: mapas/datos/indicadores_provincia.csv  (6 filas)
# =============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import _estilo as st  # noqa: E402

# Variables minimas necesarias de la base principal
COLS_CASEN = [
    "folio", "id_persona", "region", "area", "numper", "pco1_a",
    "pobreza", "pobreza_multi", "hh_d_acc",  # nivel persona/hogar
    "ysub", "ytotcorh", "yautcorh",          # ingresos del hogar
]


def media_ponderada(valores, pesos):
    """Media ponderada robusta: ignora NaN en los valores."""
    v = pd.to_numeric(valores, errors="coerce")
    p = pd.to_numeric(pesos, errors="coerce")
    mask = v.notna() & p.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=p[mask]))


def construir_indicadores():
    print("FASE 2 | ETL de indicadores CASEN 2024 por provincia")

    principal = pd.read_stata(st.CASEN_PRINCIPAL, convert_categoricals=False,
                              columns=COLS_CASEN)
    prov_comuna = pd.read_stata(st.CASEN_PROV_COMUNA, convert_categoricals=False)

    df = principal.merge(prov_comuna, on=["folio", "id_persona"], how="left")
    df = df[df["region"].isin(st.CODIGOS_REGION)].copy()
    print(f"  Personas en regiones {st.CODIGOS_REGION}: {len(df):,}")

    for c in ["pobreza", "pobreza_multi", "hh_d_acc", "ysub",
              "ytotcorh", "yautcorh", "numper", "expp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- Indicadores a NIVEL PERSONA (ponderados por expp) -------------------
    # En CASEN pobreza/area/hh_d_acc son atributos del hogar replicados en cada
    # persona; la tasa estandar es el % de personas.
    df["es_pobre_ing"] = df["pobreza"].isin([1, 2]).astype(float)
    df.loc[df["pobreza"].isna(), "es_pobre_ing"] = np.nan
    df["es_rural"] = (df["area"] == 2).astype(float)

    # --- Indicadores a NIVEL HOGAR (un registro por folio) -------------------
    # expp es constante dentro del hogar -> sirve como ponderador del hogar.
    hog = df[df["pco1_a"] == 1].drop_duplicates(subset=["folio"]).copy()
    # Los componentes de ingreso ausentes en CASEN se interpretan como cero.
    hog["ysub"] = hog["ysub"].fillna(0)
    hog["dep_subsidios"] = np.where(
        hog["ytotcorh"] > 0, hog["ysub"] / hog["ytotcorh"] * 100, np.nan)
    hog["ing_autonomo_pc"] = np.where(
        hog["numper"] > 0, hog["yautcorh"] / hog["numper"], np.nan)

    # --- Agregacion por provincia -------------------------------------------
    filas = []
    for cod, sub in df.groupby("provincia"):
        sub_hog = hog[hog["provincia"] == cod]
        poblacion = sub["expp"].sum()
        pob_salud = sub.loc[sub["hh_d_acc"] == 1, "expp"].sum()
        filas.append({
            "cod_provincia": int(cod),
            "provincia": st.PROVINCIAS[int(cod)]["nombre"],
            "region": st.PROVINCIAS[int(cod)]["region"],
            "poblacion": round(poblacion),
            "pobreza_ingreso_pct": media_ponderada(sub["es_pobre_ing"], sub["expp"]) * 100,
            "pobreza_multi_pct": media_ponderada(sub["pobreza_multi"], sub["expp"]) * 100,
            "rural_pct": media_ponderada(sub["es_rural"], sub["expp"]) * 100,
            "salud_carencia_pct": media_ponderada(sub["hh_d_acc"], sub["expp"]) * 100,
            "salud_carencia_pob": round(pob_salud),
            "dep_subsidios_pct": media_ponderada(sub_hog["dep_subsidios"], sub_hog["expp"]),
            "ing_autonomo_pc": media_ponderada(sub_hog["ing_autonomo_pc"], sub_hog["expp"]),
        })

    ind = pd.DataFrame(filas).sort_values("cod_provincia").reset_index(drop=True)

    # Brecha de dependencia de subsidios respecto al promedio de la macrozona
    # (ponderado por poblacion) -> insumo del mapa divergente.
    media_zona = np.average(ind["dep_subsidios_pct"], weights=ind["poblacion"])
    ind["dep_subsidios_brecha"] = ind["dep_subsidios_pct"] - media_zona

    assert len(ind) == 6, f"Se esperaban 6 provincias, hay {len(ind)}"
    assert ind.drop(columns=["region", "provincia"]).notna().all().all(), \
        "Hay valores nulos en la tabla de indicadores"

    ind.to_csv(st.CSV_INDICADORES, index=False, encoding="utf-8")
    print(f"  Guardado: {st.CSV_INDICADORES}")
    print(f"  Dependencia de subsidios promedio macrozona: {media_zona:.2f}%")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(ind.round(2).to_string(index=False))
    return ind


if __name__ == "__main__":
    construir_indicadores()
    print("FASE 2 completada.\n")
