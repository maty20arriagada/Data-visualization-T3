from pathlib import Path

file_path = Path(r"c:\Users\Matias Arriagada R\Documents\Universidad\Quinto año universidad\Noveno semestre\Data visualización\Tareas\Tarea 3\Entrega_anterior\entrega_final_graficos.py")
content = file_path.read_text(encoding="utf-8", errors="ignore")

lines = content.splitlines()
start = -1
for i, line in enumerate(lines):
    if "def plot_a4_g6_macrozonas" in line or "def a4_g6_macrozonas" in line or "[A4-G6]" in line:
        start = i
        break

if start != -1:
    for idx in range(start, start + 80):
        if idx < len(lines):
            print(f"{idx+1}: {lines[idx]}")
else:
    print("Function not found")
