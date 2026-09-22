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

# ============================================================
# UNIR DATASET DE FELICIDAD CON DATASET DE SALUD MENTAL
# ============================================================

# Ruta del dataset de salud mental
ruta_mental = BASE_DIR.parent / "data" / "Global_Mental_Health_Crisis_Index_2026.csv"

# Verificar que el archivo existe
if not ruta_mental.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_mental}")

# Cargar el dataset de salud mental
mental = pd.read_csv(ruta_mental)

print("\n--- Dataset de salud mental cargado ---")
print(mental.info())
print("\nPrimeras 5 filas:")
print(mental.head())

print("\nCantidad de países en felicidad:", df["country"].nunique())
print("Cantidad de países en salud mental:", mental["country"].nunique())

print("\nPaíses del dataset de felicidad:")
print(df["country"].tolist())

print("\nPaíses del dataset de salud mental:")
print(mental["country"].tolist())

mental_limpio = mental[
    [
        "country",
        "depression_pct",
        "anxiety_pct",
        "suicide_rate_per100k",
        "treatment_gap_pct",
        "youth_mh_crisis_score",
        "mh_crisis_index",
        "mh_system_score"
    ]
].copy()

df_final = pd.merge(
    df,
    mental_limpio,
    on="country",
    how="inner"
)

print("\n--- DataFrame final ---")
print(df_final.info())

print("\nPrimeras 5 filas:")
print(df_final.head())

print("\nCantidad de países antes de unir:", len(df))
print("Cantidad de países después de unir:", len(df_final))



## Dimensiones y estructura de los datos
# ============================================================
# COMPROBACIÓN DEL DATAFRAME FINAL
# ============================================================

df_final.info()
df_final.shape

# Datos faltantes
df_final.isnull().sum()  # Conteo de NA por columna
(df_final == "").sum()   # Falsos nulos (cadenas vacías)

# Duplicados
df_final.duplicated().sum()  # Cantidad de filas exactamente iguales


# ============================================================
# GRÁFICO 1: TOP 15 PAÍSES CON MAYOR PUNTAJE DE FELICIDAD
# ============================================================

# 1. Ordenar por score y quedarnos con los 15 primeros
top15 = df_final.sort_values("score", ascending=False).head(15)

# Para que el país #1 quede arriba,
# invertimos el orden antes de graficar
top15 = top15.sort_values("score", ascending=True)

# 2. fig y ax por separado
fig, ax = plt.subplots(figsize=(8, 6))

# 3. Barras horizontales
ax.barh(
    top15["country"],
    top15["score"],
    color="#4C72B0"
)

# 4. Título
ax.set_title(
    "Los 15 países con mayor puntaje de felicidad, 2026",
    fontsize=12
)

# 5. Etiquetas
ax.set_xlabel("Puntaje de felicidad (escala 0–10)")
ax.set_ylabel("")

# 6. Rango del eje X
ax.set_xlim(0, 8)

# 7. Grilla
ax.grid(axis="x", alpha=0.3)

# 8. Bordes
for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)

# 9. Fuente
fig.text(
    0.1,
    0.01,
    "Fuente: world_happiness_Update_report_2026.csv + Global_Mental_Health_Crisis_Index_2026.csv",
    fontsize=8,
    color="gray"
)

plt.tight_layout()
plt.show()


# ============================================================
# GRÁFICO 2: PBI PER CÁPITA VS. PUNTAJE DE FELICIDAD
# ============================================================

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(
    df_final["gdp_per_capita"],
    df_final["score"],
    color="#4C72B0",
    alpha=0.7,
    edgecolor="white",
    linewidth=0.5
)

ax2.set_title(
    "PBI per cápita vs. puntaje de felicidad — países con datos de ambos datasets, 2026",
    fontsize=12
)

ax2.set_xlabel("PBI per cápita (escala del índice)")
ax2.set_ylabel("Puntaje de felicidad (escala 0–10)")

ax2.grid(alpha=0.3)

for lado in ("top", "right"):
    ax2.spines[lado].set_visible(False)

fig2.text(
    0.1,
    0.01,
    "Fuente: world_happiness_Update_report_2026.csv + Global_Mental_Health_Crisis_Index_2026.csv",
    fontsize=8,
    color="gray"
)

plt.tight_layout()
plt.show()

