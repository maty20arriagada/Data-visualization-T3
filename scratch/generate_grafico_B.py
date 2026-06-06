import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

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
COLOR_TEXTO_FUERTE = "#1A2530"
COLOR_TEXTO = "#2C3E50"
A4_W, A4_H = 620, 506

# Copia de la función que acabamos de escribir en app.py
df_pobres = df_g2_macro[df_g2_macro["estado_pob"] != "Fuera de pobreza"].copy()
tabla_expr = df_pobres.groupby(["zona", "estado_pob"])["expr"].sum().unstack(fill_value=0)
tabla_expr = tabla_expr.reindex(columns=ORDEN_POBRES, fill_value=0)
tabla_expr = tabla_expr.reindex(index=ORDEN_ZONAS, fill_value=0)

sumas_zona = tabla_expr.sum(axis=1)
sumas_zona = sumas_zona.replace(0, 1)
pct = tabla_expr.div(sumas_zona, axis=0) * 100

fig = go.Figure()
for estado in ORDEN_POBRES:
    v = list(pct[estado].values)
    fig.add_trace(go.Bar(
        x=v, y=ORDEN_ZONAS, orientation="h", name=estado,
        marker=dict(color=PALETA_POBREZA[estado],
                    line=dict(color="white", width=0.6)),
        text=[f"{x:.0f}%" if x >= 9 else "" for x in v],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(family=FONT_FAMILY, size=9.5, color="white"),
        hovertemplate=(f"<b>{estado}</b><br>%{{y}}<br>"
                       "%{x:.2f}%<extra></extra>"),
    ))

def _titulo_a4(titulo: str, subtitulo: str, size_t: int = 14, size_s: int = 9.5):
    return dict(
        text=f"<b>{titulo}</b><br><sup>{subtitulo}</sup>",
        font=dict(family=FONT_FAMILY, size=size_t, color=COLOR_TEXTO_FUERTE),
        x=0.02, xanchor="left", y=0.97, yanchor="top",
        pad=dict(t=5, b=5),
    )

def _formato_n(n_muestral: int, n_ponderado: float, unidad: str = "hogares") -> str:
    return (f"n = {n_ponderado:,.0f} {unidad} (población expandida) · "
            f"{n_muestral:,} {unidad} en la muestra")

n_mues = int(df_g2_macro["n"].sum())
n_pond = float(df_g2_macro["expr"].sum())
n_label = _formato_n(n_mues, n_pond, "hogares pobres")

fig.update_layout(
    barmode="stack",
    title=_titulo_a4(
        "B. Composición de la pobreza por macrozona",
        "% de hogares pobres por tipo, normalizado al 100% intra-zona (excluye fuera de pobreza) (CASEN 2024)."
    ),
    xaxis=dict(title="% hogares pobres",
               ticksuffix="%", range=[0, 100], gridcolor="#ECF0F1",
               tickfont=dict(size=8.5),
               title_font=dict(size=9)),
    yaxis=dict(title="", autorange="reversed",
               tickfont=dict(size=9, color=COLOR_TEXTO_FUERTE)),
    legend=dict(orientation="h", yanchor="bottom", y=-0.16,
                xanchor="center", x=0.5,
                font=dict(size=8.0, family=FONT_FAMILY),
                traceorder="reversed"),
    font=dict(family=FONT_FAMILY, color=COLOR_TEXTO),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=65, b=115, l=85, r=20),
    width=A4_W, height=A4_H,
    annotations=[dict(
        x=0.5, y=-0.28, xref="paper", yref="paper",
        text=f"<i>{n_label}</i>", showarrow=False,
        xanchor="center",
        font=dict(size=7.5, color="#7F8C8D", family=FONT_FAMILY))],
)

# Guardar en la carpeta de artefactos
artifact_dir = Path(r"C:\Users\Matias Arriagada R\.gemini\antigravity-ide\brain\571e806f-1af5-4ece-a948-7756db3ae171")
fig.write_image(str(artifact_dir / "grafico_B_macrozonas_corregido.png"))
print("Saved corrected chart image to artifact folder.")
