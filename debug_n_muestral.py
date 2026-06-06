"""
Script para analizar la consistencia de 'n' en los gráficos
"""
import pandas as pd

# G3 SANKEY (Solo POBRES de Norte Grande)
g3 = pd.read_csv('processed_data/g3_sankey.csv')

print("=" * 70)
print("ANÁLISIS DE INCONSISTENCIAS EN 'n' MUESTRAL")
print("=" * 70)

print("\n1. G3 SANKEY (Gráficos D y E: Sankey y Barras)")
print("-" * 70)
print(f"   - Filtra: df_pobres[(df_pobres['zona'] == 'Norte Grande') & ...]")
print(f"   - Contiene: Solo HOGARES POBRES de Norte Grande")
print(f"   - Shape: {g3.shape[0]} filas (después de agregar)")
print(f"   - Suma de 'expr' (hogares ponderados): {g3['expr'].sum():,.0f}")
print(f"   - n_mues reportado en app.py línea 301/364: 2,613")
print(f"   - Peso promedio por hogar: {g3['expr'].sum() / 2613:.2f}")

# G4 RADAR (Todos de Norte Grande)
g4 = pd.read_csv('processed_data/g4_radar.csv')
print("\n2. G4 RADAR (Gráfico F: Radar)")
print("-" * 70)
print(f"   - Filtra: df_h[(df_h['zona'] == 'Norte Grande') & ...]")
print(f"   - Contiene: TODOS los HOGARES de Norte Grande (no solo pobres)")
print(f"   - Shape: {g4.shape[0]} filas (2 grupos: chilenos e inmigrantes)")
for idx, row in g4.iterrows():
    print(f"   - {row['origen_jefe']}: n_hog={row['n_hog']:,.0f}, n_pond={row['n_pond']:,.0f}")
print(f"   - Total n_hog: {g4['n_hog'].sum():,.0f}")
print(f"   - Total n_pond: {g4['n_pond'].sum():,.0f}")
print(f"   - n_mues reportado en app.py línea 455: 9,373")
print(f"   - Peso promedio por hogar: {g4['n_pond'].sum() / g4['n_hog'].sum():.2f}")

print("\n" + "=" * 70)
print("CONCLUSIÓN SOBRE LA CONSISTENCIA")
print("=" * 70)
print("""
Los tamaños de muestra SON DIFERENTES pero CONSISTENTES:

• Gráficos D y E (Sankey + Barras):
  - Trabajan CON: Solo pobres de Norte Grande
  - n_mues = 2,613 hogares (muestra)
  - n_ponderado = 117,569 hogares (población expandida)
  
• Gráfico F (Radar):
  - Trabaja CON: Todos los hogares de Norte Grande
  - n_mues = 9,373 hogares (muestra)  
  - n_ponderado = 450,321 hogares (población expandida)

ESTO ES CORRECTO porque son subconjuntos de datos diferentes.
Sin embargo, hay un PROBLEMA de CLARIDAD:

❌ PROBLEMA REAL:
   Los números están HARDCODEADOS (2613 y 9373) sin validación
   Si los datos fuesen re-procesados, estos números podrían quedar obsoletos
   
✓ RECOMENDACIÓN:
   Calcular dinámicamente 'n_mues' en lugar de hardcodear
   O validar que los valores hardcodeados sean correctos
""")

print("\n" + "=" * 70)
print("VERIFICACIÓN: ¿Es 2613 el número correcto?")
print("=" * 70)
print(f"""
Para verificar si 2613 es correcto, necesitaríamos conocer:
- Cuántas observaciones de hogar hay en df_ng_pob (pobres de NG)

Del cálculo inverso:
- Si 2613 hogares → 117,569 expandidos
- Peso medio: {g3['expr'].sum() / 2613:.2f}

Para estar seguro, necesitamos revisar el preprocess ejecutándolo
o agregando una columna 'n' al CSV g3_sankey.csv
""")
