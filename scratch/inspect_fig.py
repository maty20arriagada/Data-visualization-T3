import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import json

DIR_T3 = Path(r"c:\Users\Matias Arriagada R\Documents\Universidad\Quinto año universidad\Noveno semestre\Data visualización\Tareas\Tarea 3")
DIR_PROCESSED = DIR_T3 / "processed_data"

df_g2_macro = pd.read_csv(DIR_PROCESSED / "g2_macrozonas.csv")

ORDEN_POBRES = ["Pobreza por ingresos", "Pobreza multidimensional", "Pobreza ingresos y multidim."]
ORDEN_ZONAS = ["Norte Grande", "Norte Chico", "Zona Central", "Zona Sur", "Zona Austral"]
PALETA_POBREZA = {
    "Fuera de pobreza":              "#BDC3C7",  # Gris
    "Pobreza por ingresos":          "#F39D2B",  # Amarillo/Naranja
    "Pobreza multidimensional":      "#E64F42",  # Rojo/Coral
    "Pobreza ingresos y multidim.":  "#731819",  # Borgoña/Burgundy
}
FONT_FAMILY = "Inter, Roboto, sans-serif"

tab = (df_g2_macro.groupby(["zona", "estado_pob"])["expr"].sum()
       .unstack(fill_value=0)
       .reindex(columns=ORDEN_POBRES, fill_value=0)
       .reindex(index=ORDEN_ZONAS))
pct = tab.div(tab.sum(axis=1), axis=0) * 100

print("pct.index:")
print(list(pct.index))

fig = go.Figure()
for estado in ORDEN_POBRES:
    v = pct[estado].values
    print(f"Trace for {estado}: {v}")
    fig.add_trace(go.Bar(
        x=v, y=pct.index, orientation="h", name=estado,
        marker=dict(color=PALETA_POBREZA[estado],
                    line=dict(color="white", width=0.6)),
        text=[f"{x:.0f}%" if x >= 9 else "" for x in v],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(family=FONT_FAMILY, size=9.5, color="white"),
        hovertemplate=(f"<b>{estado}</b><br>%{{y}}<br>"
                       "%{x:.2f}%<extra></extra>"),
    ))

fig.update_layout(
    barmode="stack",
    xaxis=dict(title="% hogares pobres",
               ticksuffix="%", range=[0, 100]),
    yaxis=dict(title="", autorange="reversed"),
)

fig.write_image("test_bar_chart.png")
print("Saved test_bar_chart.png")
