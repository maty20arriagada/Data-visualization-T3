# =============================================================================
#  PREPROCESAMIENTO DE DATOS - Tarea 3 Data Visualization
#  Este script procesa las bases de datos de CASEN (offline) y genera los
#  archivos agregados en CSV y JSON para que el dashboard en Streamlit se cargue
#  de manera instantánea y sin problemas de memoria.
# =============================================================================
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal
import esda

warnings.filterwarnings("ignore")

# --- Rutas de datos ----------------------------------------------------------
DIR_T3 = Path(__file__).resolve().parent
DIR_T2 = DIR_T3.parent / "Tarea 2"
DIR_DATOS = DIR_T2 / "Datos"
DIR_PROCESSED = DIR_T3 / "processed_data"
DIR_PROCESSED.mkdir(exist_ok=True)

ARCHIVOS_CASEN = {
    2013: DIR_DATOS / "casen_2013.dta",
    2015: DIR_DATOS / "casen_2015.dta",
    2017: DIR_DATOS / "casen_2017.dta",
    2022: DIR_DATOS / "casen_2022.dta",
    2024: DIR_DATOS / "casen_2024.dta",
}

# --- Mapeo de regiones a macrozonas (5 zonas tradicionales) ----------------
REGION_A_ZONA = {
    15: "Norte Grande", 1: "Norte Grande", 2: "Norte Grande",
    3: "Norte Chico", 4: "Norte Chico",
    5: "Zona Central", 13: "Zona Central", 6: "Zona Central", 7: "Zona Central",
    16: "Zona Sur", 8: "Zona Sur", 9: "Zona Sur", 14: "Zona Sur", 10: "Zona Sur",
    11: "Zona Austral", 12: "Zona Austral"
}

DICC_REGIONES = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "La Araucanía",
    14: "Los Ríos", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes"
}

# --- Dimensiones Metodología 2015 (G4 Radar) --------------------------------
DIMENSIONES_MD_2015 = {
    "Educacion":              ["hh_d_asis_2015", "hh_d_rez_2015", "hh_d_esc_2015"],
    "Salud":                  ["hh_d_mal_2015", "hh_d_prevs_2015", "hh_d_acc_2015"],
    "Trabajo y Seg. Social":  ["hh_d_act_2015", "hh_d_cot_2015", "hh_d_jub_2015"],
    "Vivienda y Entorno":     ["hh_d_hacina_2015", "hh_d_estado_2015",
                               "hh_d_servbas_2015", "hh_d_entorno_2015"],
    "Redes y Cohesion":       ["hh_d_appart_2015", "hh_d_tsocial_2015", "hh_d_seg_2015"],
}

# --- Dimensiones Metodología 2024 (G10/C Sankey) ----------------------------
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


def cargar_casen_anio(anio: int, cols_pedidas: list) -> pd.DataFrame:
    archivo = ARCHIVOS_CASEN[anio]
    if not archivo.exists():
        raise FileNotFoundError(f"No se encontró {archivo}")
    
    # Obtener variables disponibles para no fallar
    reader = pd.io.stata.StataReader(archivo)
    disponibles = set(reader.variable_labels().keys())
    
    a_cargar = [c for c in cols_pedidas if c in disponibles]
    print(f"  Cargando CASEN {anio}... ({len(a_cargar)} columnas)")
    
    return pd.read_stata(archivo, convert_categoricals=False, columns=a_cargar)


def jefe_de_hogar(df: pd.DataFrame) -> pd.DataFrame:
    col = "pco1_a" if "pco1_a" in df.columns else "pco1"
    if col not in df.columns:
        # En algunas versiones viejas, el indicador de jefe es pco1 o similar
        # Validar si existe
        raise KeyError(f"No se encontró columna de jefe de hogar en {df.columns}")
    return df[df[col] == 1].drop_duplicates(subset=["folio"]).copy()


def clasificar_pobreza(es_pob_ing, es_pob_multi):
    if es_pob_ing and es_pob_multi:
        return "Pobreza ingresos y multidim."
    if es_pob_ing:
        return "Pobreza por ingresos"
    if es_pob_multi:
        return "Pobreza multidimensional"
    return "Fuera de pobreza"


def main():
    print("Iniciando preprocesamiento de datos...")
    
    # =========================================================================
    # 1. PROCESAMIENTO BASE CASEN 2024
    # =========================================================================
    # Columnas necesarias de CASEN 2024
    cols_2024 = [
        "folio", "pco1_a", "expr", "region", "area", "pobreza", "pobreza_multi", "lugar_nac", "estrato"
    ]
    
    # Agregar todas las variables de metodología 2015 y 2024
    for dim_v in DIMENSIONES_MD_2015.values():
        cols_2024.extend(dim_v)
    for dim_v in DIMENSIONES_MD_2024.values():
        cols_2024.extend(dim_v)
        
    df_raw = cargar_casen_anio(2024, cols_2024)
    df_h = jefe_de_hogar(df_raw)
    
    # Limpieza e imputación
    for c in ["expr", "region", "area", "pobreza", "pobreza_multi", "lugar_nac", "estrato"]:
        if c in df_h.columns:
            df_h[c] = pd.to_numeric(df_h[c], errors="coerce")
            
    df_h = df_h.dropna(subset=["expr"])
    
    # Excluir islas de la Zona Central (Juan Fernández = 5104, Isla de Pascua = 5201)
    if "estrato" in df_h.columns:
        df_h = df_h[~((df_h["estrato"] // 100).isin([5104, 5201]))]
    
    df_h["es_pob_ing"] = df_h["pobreza"].isin([1, 2]).astype(int)
    df_h["es_pob_multi"] = (df_h["pobreza_multi"] == 1).astype(int)
    
    df_h["estado_pob"] = df_h.apply(
        lambda r: clasificar_pobreza(r["es_pob_ing"], r["es_pob_multi"]), axis=1
    )
    df_h["es_pobre"] = (df_h["estado_pob"] != "Fuera de pobreza").astype(int)
    df_h["zona"] = df_h["region"].map(REGION_A_ZONA)
    df_h["origen_jefe"] = df_h["lugar_nac"].map({0: "Hogares chilenos", 1: "Hogares inmigrantes"})
    
    # =========================================================================
    # GRÁFICO A: Donut Nacional
    # =========================================================================
    print("Procesando Gráfico A (Donut Nacional)...")
    agg_g1 = df_h.groupby("estado_pob").agg(
        expr=("expr", "sum"),
        n=("expr", "count")
    ).reset_index()
    agg_g1.to_csv(DIR_PROCESSED / "g1_donut.csv", index=False)
    
    # =========================================================================
    # GRÁFICO B: Composición de la Pobreza por Macrozona
    # =========================================================================
    print("Procesando Gráfico B (Composición de la Pobreza por Macrozona)...")
    df_pob_macro = df_h[(df_h["es_pobre"] == 1) & df_h["zona"].notna()].copy()
    agg_macro = df_pob_macro.groupby(["zona", "estado_pob"]).agg(
        expr=("expr", "sum"),
        n=("expr", "count")
    ).reset_index()
    agg_macro.to_csv(DIR_PROCESSED / "g2_macrozonas.csv", index=False)
    
    # =========================================================================
    # MAPAS POR ZONA: Composición de la Pobreza Regional
    # =========================================================================
    print("Procesando composición regional para mapas...")
    # Solo hogares pobres
    df_pobres = df_h[df_h["es_pobre"] == 1].copy()
    
    # Calcular promedios nacionales entre los pobres para la sobrerrepresentación
    df_donut_pobres = agg_g1[agg_g1["estado_pob"] != "Fuera de pobreza"]
    total_pobres_nac = df_donut_pobres["expr"].sum()
    
    nat_ingresos = round((df_donut_pobres[df_donut_pobres["estado_pob"] == "Pobreza por ingresos"]["expr"].values[0] / total_pobres_nac) * 100, 2)
    nat_multi = round((df_donut_pobres[df_donut_pobres["estado_pob"] == "Pobreza multidimensional"]["expr"].values[0] / total_pobres_nac) * 100, 2)
    nat_ambas = round((df_donut_pobres[df_donut_pobres["estado_pob"] == "Pobreza ingresos y multidim."]["expr"].values[0] / total_pobres_nac) * 100, 2)
    
    filas_mapas = []
    for reg_code, sub in df_pobres.groupby("region"):
        tot_expr = sub["expr"].sum()
        
        pob_ing = sub[sub["estado_pob"] == "Pobreza por ingresos"]["expr"].sum()
        pob_mul = sub[sub["estado_pob"] == "Pobreza multidimensional"]["expr"].sum()
        pob_amb = sub[sub["estado_pob"] == "Pobreza ingresos y multidim."]["expr"].sum()
        
        pct_ing = (pob_ing / tot_expr * 100) if tot_expr > 0 else 0
        pct_mul = (pob_mul / tot_expr * 100) if tot_expr > 0 else 0
        pct_amb = (pob_amb / tot_expr * 100) if tot_expr > 0 else 0
        
        # Identificar la predominante
        pcts = {
            "Pobreza por ingresos": pct_ing,
            "Pobreza multidimensional": pct_mul,
            "Pobreza ingresos y multidim.": pct_amb
        }
        pred_type = max(pcts, key=pcts.get)
        pred_pct = pcts[pred_type]
        
        # Calcular tamaños de muestra de hogares por región
        sub_tot = df_h[df_h["region"] == reg_code]
        n_sample_tot = len(sub_tot)
        n_expanded_tot = sub_tot["expr"].sum()
        n_sample_poor = len(sub)
        n_expanded_poor = tot_expr
        
        # Calcular diferencias y sobrerrepresentación
        diff_ing = pct_ing - nat_ingresos
        diff_mul = pct_mul - nat_multi
        diff_amb = pct_amb - nat_ambas
        
        diffs = {
            "Pobreza por ingresos": diff_ing,
            "Pobreza multidimensional": diff_mul,
            "Pobreza ingresos y multidim.": diff_amb
        }
        cat_sobrerep = max(diffs, key=diffs.get)
        val_sobrerep = diffs[cat_sobrerep]
        
        filas_mapas.append({
            "region": int(reg_code),
            "region_name": DICC_REGIONES.get(int(reg_code), f"Región {reg_code}"),
            "zona": REGION_A_ZONA.get(int(reg_code)),
            "pct_pob_ingresos": pct_ing,
            "pct_pob_multidimensional": pct_mul,
            "pct_pob_ambas": pct_amb,
            "predominante_tipo": pred_type,
            "predominante_pct": pred_pct,
            "n_sample_tot": n_sample_tot,
            "n_expanded_tot": n_expanded_tot,
            "n_sample_poor": n_sample_poor,
            "n_expanded_poor": n_expanded_poor,
            "diff_ingresos": diff_ing,
            "diff_multi": diff_mul,
            "diff_ambas": diff_amb,
            "cat_sobrerep": cat_sobrerep,
            "val_sobrerep": val_sobrerep
        })
        
    df_regional = pd.DataFrame(filas_mapas)
    df_regional.to_csv(DIR_PROCESSED / "regional_poverty.csv", index=False)
    
    # =========================================================================
    # GRÁFICO C & D: Sankey & Barras Apiladas Norte Grande (Metodología 2024)
    # =========================================================================
    print("Procesando Gráficos C y D (Norte Grande)...")
    cols_4 = ["hh_d_defcuanti", "hh_d_esc", "hh_d_conec", "hh_d_inf"]
    cols_resto = [c for inds in DIMENSIONES_MD_2024.values() for c in inds if c not in cols_4]
    
    # Filtrar a Norte Grande y pobres
    df_ng_pob = df_pobres[(df_pobres["zona"] == "Norte Grande") & df_pobres["origen_jefe"].notna()].copy()
    
    # Sumar carencias de las 20
    cols_todas_2024 = cols_4 + cols_resto
    for c in cols_todas_2024:
        df_ng_pob[c] = pd.to_numeric(df_ng_pob[c], errors="coerce").fillna(0).astype(int)
        
    df_ng_pob["total_car_20"] = df_ng_pob[cols_todas_2024].sum(axis=1)
    
    def asignar_carencia_pred(row):
        if row["total_car_20"] == 0:
            return "Sin carencias"
        # Prioridad por ratio de la entrega anterior
        if row.get("hh_d_defcuanti") == 1:
            return "Déficit cuantitativo"
        if row.get("hh_d_esc") == 1:
            return "Escolaridad"
        if row.get("hh_d_conec") == 1:
            return "Conectividad digital"
        if row.get("hh_d_inf") == 1:
            return "Informalidad"
        return "Otras carencias"
        
    df_ng_pob["trifecta"] = df_ng_pob.apply(asignar_carencia_pred, axis=1)
    
    # Guardar agregados de flujos
    agg_sankey = df_ng_pob.groupby(["estado_pob", "origen_jefe", "trifecta"])["expr"].sum().reset_index()
    agg_sankey.to_csv(DIR_PROCESSED / "g3_sankey.csv", index=False)
    
    # =========================================================================
    # GRÁFICO E: Radar Norte Grande (Metodología 2015)
    # =========================================================================
    print("Procesando Gráfico E (Radar Norte Grande)...")
    # Norte Grande total con origen de jefe no nulo
    df_ng_radar = df_h[(df_h["zona"] == "Norte Grande") & df_h["origen_jefe"].notna()].copy()
    
    # Asegurar que las variables de carencias de 2015 sean numéricas
    for dim_v in DIMENSIONES_MD_2015.values():
        for col in dim_v:
            df_ng_radar[col] = pd.to_numeric(df_ng_radar[col], errors="coerce").fillna(0)
            
    # Calcular puntuación sumada por dimensión
    for dim, inds in DIMENSIONES_MD_2015.items():
        df_ng_radar[f"score_{dim}"] = df_ng_radar[inds].sum(axis=1)
        
    filas_radar = []
    for g in ["Hogares chilenos", "Hogares inmigrantes"]:
        sub = df_ng_radar[df_ng_radar["origen_jefe"] == g]
        n_mues = len(sub)
        n_pond = sub["expr"].sum()
        
        row_dict = {"origen_jefe": g, "n_hog": n_mues, "n_pond": n_pond}
        for dim in DIMENSIONES_MD_2015:
            # Promedio ponderado
            avg = np.average(sub[f"score_{dim}"], weights=sub["expr"]) if n_mues > 0 else 0
            row_dict[dim] = avg
        filas_radar.append(row_dict)
        
    df_radar = pd.DataFrame(filas_radar)
    df_radar.to_csv(DIR_PROCESSED / "g4_radar.csv", index=False)
    
    # =========================================================================
    # GRÁFICO F: Evolución Histórica
    # =========================================================================
    print("Procesando Gráfico F (Evolución Histórica)...")
    filas_hist = []
    for anio in ARCHIVOS_CASEN:
        df_a = cargar_casen_anio(anio, ["folio", "pco1", "pco1_a", "expr", "region", "lugar_nac"])
        df_ah = jefe_de_hogar(df_a)
        
        for c in ["expr", "region", "lugar_nac"]:
            if c in df_ah.columns:
                df_ah[c] = pd.to_numeric(df_ah[c], errors="coerce")
                
        df_ah = df_ah.dropna(subset=["expr", "region", "lugar_nac"])
        # Filtro: inmigrante = 1, chileno = 0. Otros códigos quedan fuera.
        df_ah = df_ah[df_ah["lugar_nac"].isin([0, 1])]
        df_ah["es_inmigrante"] = (df_ah["lugar_nac"] == 1).astype(int)
        df_ah["zona"] = df_ah["region"].map(REGION_A_ZONA)
        df_ah = df_ah.dropna(subset=["zona"])
        
        for z in REGION_A_ZONA.values():
            sub = df_ah[df_ah["zona"] == z]
            if len(sub) == 0:
                continue
            tasa = np.average(sub["es_inmigrante"], weights=sub["expr"]) * 100
            filas_hist.append({
                "anio": anio,
                "zona": z,
                "pct_inmig": tasa
            })
            
    df_hist = pd.DataFrame(filas_hist)
    df_hist.to_csv(DIR_PROCESSED / "g5_historical_inmig.csv", index=False)
    
    # =========================================================================
    # AUTOCORRELACIÓN ESPACIAL: Moran's I y LISA
    # =========================================================================
    print("Procesando Autocorrelación Espacial (Moran's I y LISA)...")
    # Tasa regional de pobreza multidimensional en 2024 a nivel de HOGARES
    filas_multi = []
    for reg_code, sub in df_h.groupby("region"):
        tot_expr = sub["expr"].sum()
        pob_multi_expr = sub[sub["es_pob_multi"] == 1]["expr"].sum()
        pct = (pob_multi_expr / tot_expr * 100) if tot_expr > 0 else 0
        filas_multi.append({
            "region": int(reg_code),
            "pobreza_multi_pct": pct
        })
    df_poverty_reg = pd.DataFrame(filas_multi)
    
    # Cargar geometría
    geo_path = DIR_T3 / "Regional.geojson"
    gdf = gpd.read_file(geo_path)
    gdf["codregion"] = gdf["codregion"].astype(int)
    # Excluir zona sin demarcar (código 0)
    gdf = gdf[gdf["codregion"] != 0].copy()
    
    # Alinear ordenamiento de regiones
    gdf = gdf.sort_values(by="codregion").reset_index(drop=True)
    df_poverty_reg = df_poverty_reg.sort_values(by="region").reset_index(drop=True)
    
    assert list(gdf["codregion"]) == list(df_poverty_reg["region"]), "Mapeo de regiones desalineado"
    
    # Crear matriz de pesos espaciales con K-Nearest Neighbors (k=4)
    # Chile es un país muy largo y con áreas aisladas (Magallanes y Aysén), 
    # por lo que KNN con k=4 garantiza vecindad para todas las regiones.
    w = libpysal.weights.KNN.from_dataframe(gdf, k=4)
    w.transform = 'r'  # Fila estandarizada
    
    y = df_poverty_reg["pobreza_multi_pct"].values
    
    # Estandarizar la variable
    y_mean = y.mean()
    y_std_dev = y.std()
    y_std = (y - y_mean) / y_std_dev
    
    # Calcular el rezago espacial (spatial lag)
    lag = libpysal.weights.spatial_lag.lag_spatial(w, y_std)
    
    # Calcular Moran's I global
    lm = esda.moran.Moran(y, w)
    moran_i_val = lm.I
    p_value_global = lm.p_sim
    
    # Calcular LISA local (Moran local)
    lisa = esda.moran.Moran_Local(y, w, seed=42)
    quadrants = lisa.q
    p_sims_local = lisa.p_sim
    
    # Ajuste de regresión para la línea de tendencia en el Moran scatterplot
    slope, intercept = np.polyfit(y_std, lag, 1)
    
    # Clasificación LISA y Cuadrante Moran
    df_spatial = df_poverty_reg.copy()
    df_spatial["region_name"] = df_spatial["region"].map(DICC_REGIONES)
    df_spatial["y_std"] = y_std
    df_spatial["spatial_lag"] = lag
    
    # Cuadrante Moran
    def get_quadrant(y_s, lg):
        if y_s > 0 and lg > 0: return "HH"
        if y_s < 0 and lg > 0: return "LH"
        if y_s < 0 and lg < 0: return "LL"
        return "HL"
        
    df_spatial["quadrant"] = df_spatial.apply(lambda r: get_quadrant(r["y_std"], r["spatial_lag"]), axis=1)
    
    # LISA Cluster Class
    lisa_classes = []
    for idx, row in df_spatial.iterrows():
        p_val = p_sims_local[idx]
        q = quadrants[idx]
        if p_val >= 0.05:
            lisa_classes.append("No significativo")
        else:
            if q == 1: lisa_classes.append("High-High")
            elif q == 2: lisa_classes.append("Low-High")
            elif q == 3: lisa_classes.append("Low-Low")
            elif q == 4: lisa_classes.append("High-Low")
            
    df_spatial["lisa_class"] = lisa_classes
    df_spatial["lisa_p_value"] = p_sims_local
    
    # Guardar datos espaciales
    df_spatial.to_csv(DIR_PROCESSED / "moran_lisa.csv", index=False)
    
    # Guardar estadísticas globales de Moran
    stats = {
        "moran_i": float(moran_i_val),
        "p_value": float(p_value_global),
        "slope": float(slope),
        "intercept": float(intercept)
    }
    with open(DIR_PROCESSED / "moran_stats.json", "w") as f:
        json.dump(stats, f)
        
    print("Preprocesamiento finalizado con éxito. Todos los archivos auxiliares generados en:")
    print(f"  {DIR_PROCESSED}")


if __name__ == "__main__":
    main()
