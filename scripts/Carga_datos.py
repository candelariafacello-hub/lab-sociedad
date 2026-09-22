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

# Promediar países repetidos
df = df.groupby("country", as_index=False).mean(numeric_only=True)

print("Filas:", len(df))
print("Países únicos:", df["country"].nunique())
print("Duplicados:", df["country"].duplicated().sum())

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

print("Filas totales:", len(df_final))
print("Países únicos:", df_final["country"].nunique())
print("Duplicados por país:", df_final["country"].duplicated().sum())
print(df_final["country"].value_counts()[df_final["country"].value_counts() > 1])

print(
    df_final[df_final["country"].isin(
        ["Lebanon", "Mozambique", "Malawi", "Zimbabwe"]
    )].sort_values("country")
)
print(
    df[df["country"].isin(
        ["Lebanon", "Mozambique", "Malawi", "Zimbabwe"]
    )][
        ["country", "rank", "gdp_per_capita",
         "social_support", "healthy_life_expectancy",
         "freedom", "generosity", "corruption", "score"]
    ].sort_values("country")
)
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


# ============================================================
# GRÁFICO 3: CORRELACIÓN DE LAS VARIABLES CON LA FELICIDAD
# ============================================================

variables = [
    "gdp_per_capita",
    "social_support",
    "healthy_life_expectancy",
    "freedom",
    "generosity",
    "corruption"
]

# Calcular correlación de cada variable con score
correlaciones = df_final[variables + ["score"]].corr()["score"].drop("score")

print("\nCorrelaciones con la felicidad:")
print(correlaciones)

# Ordenar de menor a mayor
correlaciones = correlaciones.sort_values()

fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.barh(
    correlaciones.index,
    correlaciones.values
)

ax3.set_title(
    "Correlación de distintas variables con el puntaje de felicidad",
    fontsize=12
)

ax3.set_xlabel("Correlación con score")
ax3.set_ylabel("")

ax3.axvline(0, linewidth=1)
ax3.grid(axis="x", alpha=0.3)

for lado in ("top", "right", "left"):
    ax3.spines[lado].set_visible(False)

plt.tight_layout()

fig3.savefig(
    BASE_DIR.parent / "graficos" / "correlacion_variables_felicidad.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Cuanto mas cerca de uno mas fuerte es la relacion positiva con felicidad

# ============================================================
# GRÁFICO 4: TOP 10 VS BOTTOM 10
# ============================================================

# Seleccionar los 10 países más felices y los 10 menos felices
top10 = df_final.nlargest(10, "score")
bottom10 = df_final.nsmallest(10, "score")

# Promedio de las variables en cada grupo
promedio_top10 = top10[variables].mean()
promedio_bottom10 = bottom10[variables].mean()

print("\nPromedio TOP 10:")
print(promedio_top10)

print("\nPromedio BOTTOM 10:")
print(promedio_bottom10)

# Crear tabla para graficar
comparacion = pd.DataFrame({
    "10 países con mayor felicidad": promedio_top10,
    "10 países con menor felicidad": promedio_bottom10
})

fig4, ax4 = plt.subplots(figsize=(10, 6))

comparacion.plot(
    kind="bar",
    ax=ax4
)

ax4.set_title(
    "Comparación de variables: 10 países con mayor y menor felicidad",
    fontsize=12
)

ax4.set_xlabel("")
ax4.set_ylabel("Promedio de la variable")

ax4.grid(axis="y", alpha=0.3)

plt.xticks(rotation=35, ha="right")

plt.tight_layout()

fig4.savefig(
    BASE_DIR.parent / "graficos" / "top10_vs_bottom10_felicidad.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()