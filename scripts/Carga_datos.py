import os 
os.getcwd()

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Ruta ancla: la carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent

# Subís desde scripts/ hasta la raíz del repo, y bajás a data/
ruta_csv = BASE_DIR.parent / "data" / "world_happiness_Update_report_2026.csv"

# Verificar que el archivo realmente existe antes de cargarlo
if not ruta_csv.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv}")

# Cargar el archivo CSV
df = pd.read_csv(ruta_csv)

# Inspección inicial de los datos
print(f"--- Archivo cargado exitosamente desde: {ruta_csv.name} ---")
print(df.info())
print("\nPrimeras 5 filas:")
print(df.head())

## Dimensiones y estructura de los datos
df.info()
df.shape

# Datos faltantes
df.isnull().sum()  # Conteo de NA por columna
(df == "").sum()   # Falsos nulos (cadenas vacías)

##Duplicados
df.duplicated().sum() # Cantidad de filas exactamente iguales
# %%

# Gráfico: Top 15 países con mayor puntaje de felicidad

# 1. Preparar los datos: ordenar por score y quedarnos con los 15 primeros
top15 = df.sort_values("score", ascending=False).head(15)

# Para que en el gráfico horizontal el país #1 quede arriba,
# invertimos el orden antes de graficar
top15 = top15.sort_values("score", ascending=True)

# 2. fig y ax por separado (no plt.plot suelto)
fig, ax = plt.subplots(figsize=(8, 6))

# 3. Geometría: barras horizontales (barh), un solo atributo estético (color)
ax.barh(top15["country"], top15["score"], color="#4C72B0")

# 4. Título: qué, de quién (universo) y cuándo
ax.set_title("Los 15 países con mayor puntaje de felicidad, 2026",
             fontsize=12)

# 5. Etiquetas de eje: variable + unidad explícita
ax.set_xlabel("Puntaje de felicidad (escala 0–10)")
ax.set_ylabel("")  # los nombres de país ya identifican cada barra, no hace falta repetir "País"

# 6. Marcas: fijamos el rango del eje X para no exagerar diferencias chicas
ax.set_xlim(0, 8)

# 7. Grilla sutil, solo en el eje que se está leyendo (X), detrás de los datos
ax.grid(axis="x", alpha=0.3)

# 8. Bordes: sacamos los que no aportan referencia
for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)

# 9. Nota al pie: fuente de los datos
fig.text(0.1, 0.01,
         "Fuente: world_happiness_Update_report_2026.csv",
         fontsize=8, color="gray")

plt.tight_layout()
plt.show()

# --- Gráfico 2: dispersión PBI per cápita vs. puntaje de felicidad, todos los países ---

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(df["gdp_per_capita"], df["score"],
            color="#4C72B0", alpha=0.7, edgecolor="white", linewidth=0.5)

ax2.set_title("PBI per cápita vs. puntaje de felicidad — 147 países, 2026",
              fontsize=12)
ax2.set_xlabel("PBI per cápita (escala del índice)")
ax2.set_ylabel("Puntaje de felicidad (escala 0–10)")

ax2.grid(alpha=0.3)
for lado in ("top", "right"):
    ax2.spines[lado].set_visible(False)

fig2.text(0.1, 0.01,
          "Fuente: world_happiness_Update_report_2026.csv",
          fontsize=8, color="gray")

plt.tight_layout()
plt.show()