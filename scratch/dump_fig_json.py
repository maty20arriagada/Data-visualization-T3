import pandas as pd
from pathlib import Path
import json

DIR_T3 = Path(r"c:\Users\Matias Arriagada R\Documents\Universidad\Quinto año universidad\Noveno semestre\Data visualización\Tareas\Tarea 3")
DIR_PROCESSED = DIR_T3 / "processed_data"

df_g2_macro = pd.read_csv(DIR_PROCESSED / "g2_macrozonas.csv")

ORDEN_POBRES = ["Pobreza por ingresos", "Pobreza multidimensional", "Pobreza ingresos y multidim."]
ORDEN_ZONAS = ["Norte Grande", "Norte Chico", "Zona Central", "Zona Sur", "Zona Austral"]

tab = (df_g2_macro.groupby(["zona", "estado_pob"])["expr"].sum()
       .unstack(fill_value=0)
       .reindex(columns=ORDEN_POBRES, fill_value=0)
       .reindex(index=ORDEN_ZONAS))
pct = tab.div(tab.sum(axis=1), axis=0) * 100

import plotly.graph_objects as go
fig = go.Figure()
for estado in ORDEN_POBRES:
    v = pct[estado].values
    fig.add_trace(go.Bar(
        x=v, y=pct.index, orientation="h", name=estado,
    ))

print(json.dumps(fig.to_dict(), indent=2))
