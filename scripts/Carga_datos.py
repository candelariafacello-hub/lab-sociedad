import os 
os.getcwd()

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta ancla: carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent

# Ruta del dataset del World Happiness Report
ruta_csv_happiness = BASE_DIR.parent / "data" / "world_happiness_Update_report_2026.csv"

# Verificar que el archivo realmente existe antes de cargarlo
if not ruta_csv_happiness.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv_happiness}")

# Cargar el archivo CSV
df_happiness = pd.read_csv(ruta_csv_happiness)

print(f"--- Archivo cargado exitosamente desde: {ruta_csv_happiness.name} ---")

# ============================================================
# LIMPIEZA DE DATOS: Valores duplicados
# ============================================================

print("\nInfo del dataset original del World Happiness Report")
print("Filas:", len(df_happiness))
print("Países únicos:", df_happiness["country"].nunique())
print("Duplicados:", df_happiness["country"].duplicated().sum()) # Problema: Paises duplicados
print(df_happiness.info()) # Vemos que no hay nulos

# Promediar países repetidos
df_happiness_limpio = df_happiness.groupby("country", as_index=False).mean(numeric_only=True)

print("\nInfo del dataset del World Happiness Report limpio sin duplicados")
print("Filas:", len(df_happiness_limpio))
print("Países únicos:", df_happiness_limpio["country"].nunique())
print("Duplicados:", df_happiness_limpio["country"].duplicated().sum())
print(df_happiness_limpio.info())

print("\nPrimeras 5 filas:")
print(df_happiness_limpio.head())

# ============================================================
# UNIR DATASET DE FELICIDAD CON DATASET DE SALUD MENTAL
# ============================================================

# Ruta del dataset de salud mental
ruta_csv_mental = BASE_DIR.parent / "data" / "Global_Mental_Health_Crisis_Index_2026.csv"

# Verificar que el archivo existe
if not ruta_csv_mental.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv_mental}")

# Cargar el dataset de salud mental
mental = pd.read_csv(ruta_csv_mental)

print("\n--- Dataset de salud mental cargado ---")
print(mental.info()) # No hay valores nulos
print("\nPrimeras 5 filas:")
print(mental.head())

# Selecciono las variables de interes
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

print("\nDataset de salud mental limpio: conservamos solo variables pertinentes")
print(mental_limpio.head())

print("\nCantidad de países en el dataset de felicidad:", df_happiness_limpio["country"].nunique())
print("Cantidad de países en el dataset de salud mental:", mental["country"].nunique()) # menos
print("Nota: Al unir los dos datasets se van a perder los paises sin registros en el dataset de salud mental")

# Uno el dataset de felicidad con el de salud mental por pais
df_final = pd.merge(
    df_happiness_limpio,
    mental_limpio,
    on="country",
    how="inner" # para que no se repita la columna de pais 
)

print("\n--- DataFrame final ---")
print(df_final.info())
df_final.shape

print("\nPrimeras 5 filas:")
print(df_final.head())

print("\nCantidad de países antes de unir:", len(df_happiness_limpio))
print("Cantidad de países después de unir:", len(df_final))

print("Filas totales:", len(df_final))
print("Países únicos:", df_final["country"].nunique())
print("Duplicados por país:", df_final["country"].duplicated().sum()) # Dataset limpio sin duplicados
print(df_final["country"].value_counts()[df_final["country"].value_counts() > 1])

# ==================================================================
# ESTUDIO DE CORRELACIONES ENTRE VARIABLES Y EL SCORE DE FELICIDAD
# ==================================================================

# GRÁFICOS DE CORRELACIÓN: 

def grafico_correlacion(df, var_x, var_y='score', color='#4C72B0'):
    '''
        Crea un gráfico de correlación en base a los datos de un df, una variable que vas cambiando y la variable score. 
        Muestra:
        - una recta de regresión lineal simple (ajuste por mínimos cuadrados)
        - el coeficiente de correlación de Pearson entre ambas variables
        - color default que vamos a usar para el World Happiness Report sino (para salud mental) se puede especificar otro
        '''
    x = df[var_x]
    y = df[var_y]
    
    # Correlación
    corr = np.corrcoef(x, y)[0, 1]
    
    # Ajuste lineal (regresión)
    coef = np.polyfit(x, y, 1)
    recta = np.poly1d(coef)
    x_line = np.linspace(x.min(), x.max(), 100)
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, color=color, alpha=0.75, edgecolor='white', linewidth=0.5, s=40)
    ax.plot(x_line, recta(x_line), color='#D62728', linewidth=2)
    
    ax.text(0.05, 0.92, f'Correlación: {corr:.2f}',
            transform=ax.transAxes, fontsize=10, color='#333333')
    
    # Título representativo
    titulo = f'{var_x.replace("_", " ").title()} vs {var_y.title()}'
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12)
    
    ax.set_xlabel(var_x.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel(var_y.title(), fontsize=10)
    
    # estética general
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR.parent / "graficos" / f'correlacion_{var_x}.png', dpi=300, bbox_inches="tight")
    plt.show()


# 1) VARIABLES DEL WORLD HAPPINESS REPORT:
# Como el df_happiness_limpio tiene más paises que el df_final vamos a usar este primero para el estudio de las 
# variables puntuales para que sea mas representativo. 

grafico_correlacion(df_happiness_limpio, 'gdp_per_capita')
grafico_correlacion(df_happiness_limpio, 'social_support')
grafico_correlacion(df_happiness_limpio, 'healthy_life_expectancy')
grafico_correlacion(df_happiness_limpio, 'freedom')
grafico_correlacion(df_happiness_limpio, 'generosity')
grafico_correlacion(df_happiness_limpio, 'corruption')

# 2) VARIABLES DE SALUD MENTAL:
# Para el estudio de las variables de salud mental si usamos el df_final que las incluye y las une con el score de felicidad
grafico_correlacion(df_final, 'depression_pct', color='#55A868')
grafico_correlacion(df_final, 'suicide_rate_per100k', color='#55A868')
grafico_correlacion(df_final, 'treatment_gap_pct', color='#55A868')
grafico_correlacion(df_final, 'youth_mh_crisis_score', color='#55A868')
grafico_correlacion(df_final, 'mh_crisis_index', color='#55A868')
grafico_correlacion(df_final, 'mh_system_score', color='#55A868')


# GRÁFICO DE BARRAS DE RESUMEN: Puntaje de correlación de las variables del World Happiness Report

variables = [
    "gdp_per_capita",
    "social_support",
    "healthy_life_expectancy",
    "freedom",
    "generosity",
    "corruption"
]

# Calcular correlación de cada variable con score
correlaciones = df_happiness_limpio[variables + ["score"]].corr()["score"].drop("score")
print("\nCorrelaciones de variables del World Happiness Report con el puntaje de felicidad:")
print(correlaciones)

# Ordenar de menor a mayor
correlaciones = correlaciones.sort_values()

# Gráfico
fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.barh(correlaciones.index, correlaciones.values, color='#4C72B0')

ax3.set_title("Correlación de variables del World Happiness Report con el puntaje de felicidad", fontsize=13, fontweight='bold')

ax3.set_xlabel("Correlación con score")
ax3.set_ylabel("")

ax3.axvline(0, linewidth=1)
ax3.grid(axis="x", alpha=0.3)

for lado in ("top", "right", "left"):
    ax3.spines[lado].set_visible(False)

plt.tight_layout()
fig3.savefig(BASE_DIR.parent / "graficos" / "resumen_correlacion_variables_world_happiness_report.png", dpi=300, bbox_inches="tight")
plt.show()


# GRÁFICO DE BARRAS DE RESUMEN: Puntaje de correlación de las variables de salud mental 

variables = [
    "depression_pct",
    "anxiety_pct",
    "suicide_rate_per100k",
    "treatment_gap_pct",
    "youth_mh_crisis_score",
    "mh_crisis_index",
    "mh_system_score"
]

# Calcular correlación de cada variable con score
correlaciones = df_final[variables + ["score"]].corr()["score"].drop("score")
print("\nCorrelaciones de variables de salud mental con el puntaje de felicidad:")
print(correlaciones)

# Ordenar de menor a mayor
correlaciones = correlaciones.sort_values()

# Gráfico
fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.barh(correlaciones.index, correlaciones.values, color='#55A868')

ax3.set_title("Correlación de variables de salud mental con el puntaje de felicidad", fontsize=13, fontweight='bold')

ax3.set_xlabel("Correlación con score")
ax3.set_ylabel("")

ax3.axvline(0, linewidth=1)
ax3.grid(axis="x", alpha=0.3)

for lado in ("top", "right", "left"):
    ax3.spines[lado].set_visible(False)

plt.tight_layout()
fig3.savefig(BASE_DIR.parent / "graficos" / "resumen_correlacion_variables_salud_mental.png", dpi=300, bbox_inches="tight")
plt.show()

# ====================================================
# PROMEDIOS DE VARIABLES EN PAISES TOP Y BOTTOM 10
# ====================================================

# Gráfico de barras
variables_happiness = [
    "gdp_per_capita", "social_support", "healthy_life_expectancy",
    "freedom", "generosity", "corruption"
]

variables_salud_mental = [
    "depression_pct", "anxiety_pct", "suicide_rate_per100k",
    "treatment_gap_pct", "youth_mh_crisis_score", "mh_crisis_index", "mh_system_score"
]

variables_todas = variables_happiness + variables_salud_mental

# Seleccionar los 10 países más felices y los 10 menos felices
top10 = df_final.nlargest(10, "score")
bottom10 = df_final.nsmallest(10, "score")

# Promedio de las variables en cada grupo
promedio_top10 = top10[variables_todas].mean()
promedio_bottom10 = bottom10[variables_todas].mean()

print("\nPromedio TOP 10:")
print(promedio_top10)

print("\nPromedio BOTTOM 10:")
print(promedio_bottom10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

comparacion_hap = pd.DataFrame({
    "10 países con mayor felicidad": promedio_top10[variables_happiness],
    "10 países con menor felicidad": promedio_bottom10[variables_happiness]
})
comparacion_hap.plot(kind="bar", ax=ax1, color=['#E8A33D', '#8C6BAE'])
ax1.set_title("Variables del World Happiness Report", fontsize=12, fontweight='bold')
ax1.set_ylabel("Promedio")
ax1.grid(axis="y", alpha=0.3)
ax1.tick_params(axis='x', rotation=35)

comparacion_sm = pd.DataFrame({
    "10 países con mayor felicidad": promedio_top10[variables_salud_mental],
    "10 países con menor felicidad": promedio_bottom10[variables_salud_mental]
})
comparacion_sm.plot(kind="bar", ax=ax2, color=['#E8A33D', '#8C6BAE'])
ax2.set_title("Variables de salud mental", fontsize=12, fontweight='bold')
ax2.set_ylabel("Promedio")
ax2.grid(axis="y", alpha=0.3)
ax2.tick_params(axis='x', rotation=35)

for ax in (ax1, ax2):
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.suptitle("Promedio de variables: 10 países con mayor y menor felicidad", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(BASE_DIR.parent / "graficos" / "top10_vs_bottom10_felicidad.png", dpi=300, bbox_inches="tight")
plt.show()

import plotly.express as px
import plotly.io as pio
import pycountry_convert as pc

pio.renderers.default = "browser"   # abre el mapa en el navegador (necesario en Spyder)

# 1) Región de cada país (continente)
correcciones = {
    "Kosovo": "Europa", "Taiwan Province of China": "Asia",
    "Hong Kong S.A.R. of China": "Asia", "Palestinian Territories": "Asia",
    "Northern Cyprus": "Europa", "North Cyprus": "Europa",
    "Somaliland region": "África", "Somaliland Region": "África",
    "Congo (Kinshasa)": "África", "Congo (Brazzaville)": "África",
    "Türkiye": "Asia", "Turkiye": "Asia",
}
continentes = {"AF": "África", "AS": "Asia", "EU": "Europa",
               "NA": "América del Norte", "SA": "América del Sur", "OC": "Oceanía"}

def obtener_region(pais):
    if pais in correcciones:
        return correcciones[pais]
    try:
        cod = pc.country_name_to_country_alpha2(pais)
        return continentes[pc.country_alpha2_to_continent_code(cod)]
    except Exception:
        return None

df_final["region"] = df_final["country"].apply(obtener_region)
print("Países sin región:", df_final.loc[df_final["region"].isna(), "country"].tolist())
df_final = df_final.dropna(subset=["region"])

# 2) Promedios por región
vars_region = ["score", "depression_pct", "anxiety_pct", "mh_crisis_index"]
promedios = df_final.groupby("region")[vars_region].mean().round(2)
print(promedios)

# 3) Cada país toma los valores promedio de su región
mapa_df = df_final[["country", "region"]].merge(promedios, on="region")

# 4) Mapa: color = felicidad promedio de la región
fig = px.choropleth(
    mapa_df,
    locations="country",
    locationmode="country names",
    color="score",
    hover_name="region",
    hover_data={"country": False, "score": ":.2f", "depression_pct": ":.2f",
                "anxiety_pct": ":.2f", "mh_crisis_index": ":.2f"},
    color_continuous_scale="YlGnBu",
    labels={"score": "Felicidad", "depression_pct": "Depresión %",
            "anxiety_pct": "Ansiedad %", "mh_crisis_index": "Índice de crisis"},
    title="Felicidad y salud mental promedio por región"
)
fig.update_layout(geo=dict(showframe=False, showcoastlines=False,
                           projection_type="natural earth"))
fig.write_html(BASE_DIR.parent / "graficos" / "mapa_regiones.html")  # solo HTML, sin kaleido
fig.show()