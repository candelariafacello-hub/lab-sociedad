import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pycountry_convert as pc
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# NOMBRES DE COLUMNAS EN ESPAÑOL
# ============================================================
# Se aplican apenas se carga cada dataset (antes del merge), así todo el
# resto del script usa los nombres nuevos.

NOMBRES_ES = {
    "country": "País",
    "rank": "Ranking",
    "gdp_per_capita": "PIB per cápita",
    "social_support": "Apoyo social",
    "healthy_life_expectancy": "Esperanza de vida saludable",
    "freedom": "Libertad",
    "generosity": "Generosidad",
    "corruption": "Percepción de corrupción",
    "score": "Puntaje de felicidad",
    "depression_pct": "Depresión (%)",
    "anxiety_pct": "Ansiedad (%)",
    "suicide_rate_per100k": "Suicidios cada 100k hab.",
    "treatment_gap_pct": "Brecha de tratamiento (%)",
    "youth_mh_crisis_score": "Crisis de salud mental juvenil",
    "mh_crisis_index": "Índice de crisis de salud mental",
    "mh_system_score": "Calidad del sistema de salud mental",
}

# Atajos para no repetir textos largos
PAIS = NOMBRES_ES["country"]
PUNTAJE = NOMBRES_ES["score"]
REGION = "Región"


def slug(texto):
    """Convierte un nombre a algo seguro para nombre de archivo (sin tildes ni espacios)."""
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


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
print("Duplicados:", df_happiness["country"].duplicated().sum())  # Problema: Paises duplicados
print(df_happiness.info())  # Vemos que no hay nulos

# Promediar países repetidos
df_happiness_limpio = df_happiness.groupby("country", as_index=False).mean(numeric_only=True)

# Renombrar columnas al español
df_happiness_limpio = df_happiness_limpio.rename(columns=NOMBRES_ES)

print("\nInfo del dataset del World Happiness Report limpio sin duplicados")
print("Filas:", len(df_happiness_limpio))
print("Países únicos:", df_happiness_limpio[PAIS].nunique())
print("Duplicados:", df_happiness_limpio[PAIS].duplicated().sum())
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
print(mental.info())  # No hay valores nulos
print("\nPrimeras 5 filas:")
print(mental.head())

# Selecciono las variables de interes (con los nombres originales del CSV)
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

# Renombrar columnas al español
mental_limpio = mental_limpio.rename(columns=NOMBRES_ES)

print("\nDataset de salud mental limpio: conservamos solo variables pertinentes")
print(mental_limpio.head())

print("\nCantidad de países en el dataset de felicidad:", df_happiness_limpio[PAIS].nunique())
print("Cantidad de países en el dataset de salud mental:", mental["country"].nunique())  # menos
print("Nota: Al unir los dos datasets se van a perder los paises sin registros en el dataset de salud mental")

# Uno el dataset de felicidad con el de salud mental por pais
df_final = pd.merge(
    df_happiness_limpio,
    mental_limpio,
    on=PAIS,
    how="inner"  # para que no se repita la columna de pais
)

print("\n--- DataFrame final ---")
print(df_final.info())
df_final.shape

print("\nPrimeras 5 filas:")
print(df_final.head())

print("\nCantidad de países antes de unir:", len(df_happiness_limpio))
print("Cantidad de países después de unir:", len(df_final))

print("Filas totales:", len(df_final))
print("Países únicos:", df_final[PAIS].nunique())
print("Duplicados por país:", df_final[PAIS].duplicated().sum())  # Dataset limpio sin duplicados
print(df_final[PAIS].value_counts()[df_final[PAIS].value_counts() > 1])

# ==================================================================
# ESTUDIO DE CORRELACIONES ENTRE VARIABLES Y EL PUNTAJE DE FELICIDAD
# ==================================================================

# GRÁFICOS DE CORRELACIÓN:
def grafico_correlacion(df, var_x, var_y=PUNTAJE, color='#4C72B0', n_atipicos=5):
    '''
    Crea un gráfico de correlación entre var_x y var_y (por defecto el puntaje de felicidad).
    Muestra:
    - una recta de regresión lineal simple (ajuste por mínimos cuadrados)
    - el coeficiente de correlación de Pearson
    - los n_atipicos países más alejados de la recta (mayor residuo en valor
      absoluto), etiquetados con su nombre. Se calcula solo para cada variable.
    '''
    # Quedarse solo con filas completas para x e y (evita problemas con NaN)
    datos = df[[PAIS, var_x, var_y]].dropna()
    x = datos[var_x]
    y = datos[var_y]

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

    # Países atípicos: mayor distancia vertical a la recta
    if n_atipicos > 0:
        residuos = y - recta(x)
        idx_atipicos = residuos.abs().nlargest(n_atipicos).index
        atipicos = datos.loc[idx_atipicos]

        x_rango = x.max() - x.min()
        for _, pais in atipicos.iterrows():
            # Si está en la mitad derecha del gráfico, la etiqueta va a la izquierda
            a_la_derecha = pais[var_x] > x.min() + 0.7 * x_rango
            # Si el punto está por encima de la recta, etiqueta arriba; si no, abajo
            arriba = (pais[var_y] - recta(pais[var_x])) > 0
            dx = -8 if a_la_derecha else 8
            dy = 10 if arriba else -14
            ax.annotate(
                pais[PAIS],
                (pais[var_x], pais[var_y]),
                xytext=(dx, dy),
                textcoords='offset points',
                ha='right' if a_la_derecha else 'left',
                fontsize=8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5)
            )

    # Título representativo (los nombres de columna ya están en español)
    ax.set_title(f'{var_x} vs {var_y}', fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel(var_x, fontsize=10)
    ax.set_ylabel(var_y, fontsize=10)

    # Estética general
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(BASE_DIR.parent / "graficos" / f'correlacion_{slug(var_x)}.png',
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# 1) VARIABLES DEL WORLD HAPPINESS REPORT:
# Como el df_happiness_limpio tiene más paises que el df_final vamos a usar este primero para el estudio de las
# variables puntuales para que sea mas representativo.

grafico_correlacion(df_happiness_limpio, 'PIB per cápita')
grafico_correlacion(df_happiness_limpio, 'Apoyo social')
grafico_correlacion(df_happiness_limpio, 'Esperanza de vida saludable')
grafico_correlacion(df_happiness_limpio, 'Libertad')
grafico_correlacion(df_happiness_limpio, 'Generosidad')
grafico_correlacion(df_happiness_limpio, 'Percepción de corrupción')

# 2) VARIABLES DE SALUD MENTAL:
# Para el estudio de las variables de salud mental si usamos el df_final que las incluye y las une con el puntaje de felicidad
grafico_correlacion(df_final, 'Depresión (%)', color='#55A868')
grafico_correlacion(df_final, 'Suicidios cada 100k hab.', color='#55A868')
grafico_correlacion(df_final, 'Brecha de tratamiento (%)', color='#55A868')
grafico_correlacion(df_final, 'Crisis de salud mental juvenil', color='#55A868')
grafico_correlacion(df_final, 'Índice de crisis de salud mental', color='#55A868')
grafico_correlacion(df_final, 'Calidad del sistema de salud mental', color='#55A868')


# GRÁFICO DE BARRAS DE RESUMEN: Correlación de las variables del World Happiness Report

variables = [
    "PIB per cápita",
    "Apoyo social",
    "Esperanza de vida saludable",
    "Libertad",
    "Generosidad",
    "Percepción de corrupción"
]

# Calcular correlación de cada variable con el puntaje
correlaciones = df_happiness_limpio[variables + [PUNTAJE]].corr()[PUNTAJE].drop(PUNTAJE)
print("\nCorrelaciones de variables del World Happiness Report con el puntaje de felicidad:")
print(correlaciones)

# Ordenar de menor a mayor
correlaciones = correlaciones.sort_values()

# Gráfico
fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.barh(correlaciones.index, correlaciones.values, color='#4C72B0')

ax3.set_title("Correlación de variables del World Happiness Report con el puntaje de felicidad", fontsize=13, fontweight='bold')

ax3.set_xlabel("Correlación con el puntaje de felicidad")
ax3.set_ylabel("")

ax3.axvline(0, linewidth=1)
ax3.grid(axis="x", alpha=0.3)

for lado in ("top", "right", "left"):
    ax3.spines[lado].set_visible(False)

plt.tight_layout()
fig3.savefig(BASE_DIR.parent / "graficos" / "resumen_correlacion_variables_world_happiness_report.png", dpi=300, bbox_inches="tight")
plt.close(fig3)


# GRÁFICO DE BARRAS DE RESUMEN: Correlación de las variables de salud mental

variables = [
    "Depresión (%)",
    "Ansiedad (%)",
    "Suicidios cada 100k hab.",
    "Brecha de tratamiento (%)",
    "Crisis de salud mental juvenil",
    "Índice de crisis de salud mental",
    "Calidad del sistema de salud mental"
]

# Calcular correlación de cada variable con el puntaje
correlaciones = df_final[variables + [PUNTAJE]].corr()[PUNTAJE].drop(PUNTAJE)
print("\nCorrelaciones de variables de salud mental con el puntaje de felicidad:")
print(correlaciones)

# Ordenar de menor a mayor
correlaciones = correlaciones.sort_values()

# Gráfico
fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.barh(correlaciones.index, correlaciones.values, color='#55A868')

ax3.set_title("Correlación de variables de salud mental con el puntaje de felicidad", fontsize=13, fontweight='bold')

ax3.set_xlabel("Correlación con el puntaje de felicidad")
ax3.set_ylabel("")

ax3.axvline(0, linewidth=1)
ax3.grid(axis="x", alpha=0.3)

for lado in ("top", "right", "left"):
    ax3.spines[lado].set_visible(False)

plt.tight_layout()
fig3.savefig(BASE_DIR.parent / "graficos" / "resumen_correlacion_variables_salud_mental.png", dpi=300, bbox_inches="tight")
plt.close(fig3)

# ====================================================
# PROMEDIOS DE VARIABLES EN PAISES TOP Y BOTTOM 10
# ====================================================

variables_happiness = [
    "PIB per cápita", "Apoyo social", "Esperanza de vida saludable",
    "Libertad", "Generosidad", "Percepción de corrupción"
]

variables_salud_mental = [
    "Depresión (%)", "Ansiedad (%)", "Suicidios cada 100k hab.",
    "Brecha de tratamiento (%)", "Crisis de salud mental juvenil",
    "Índice de crisis de salud mental", "Calidad del sistema de salud mental"
]

variables_todas = variables_happiness + variables_salud_mental

# Seleccionar los 10 países más felices y los 10 menos felices
top10 = df_final.nlargest(10, PUNTAJE)
bottom10 = df_final.nsmallest(10, PUNTAJE)

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
plt.close(fig)

# ====================================================
# MAPA FELICIDAD Y SALUD MENTAL PROMEDIO POR REGION
# ====================================================

pio.renderers.default = "browser"   # abre el mapa en el navegador (necesario en Spyder)

# 1) Región de cada país (continente)
# OJO: pycountry_convert necesita los nombres de país en inglés (como vienen en el CSV)
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

df_final[REGION] = df_final[PAIS].apply(obtener_region)
print("Países sin región:", df_final.loc[df_final[REGION].isna(), PAIS].tolist())
df_final = df_final.dropna(subset=[REGION])

# 2) Promedios por región
vars_region = [PUNTAJE, "Depresión (%)", "Ansiedad (%)", "Índice de crisis de salud mental"]
promedios = df_final.groupby(REGION)[vars_region].mean().round(2)
print(promedios)

# 3) Cada país toma los valores promedio de su región
mapa_df = df_final[[PAIS, REGION]].merge(promedios, on=REGION)

# 4) Mapa: color = felicidad promedio de la región
fig = px.choropleth(
    mapa_df,
    locations=PAIS,
    locationmode="country names",
    color=PUNTAJE,
    hover_name=REGION,
    hover_data={PAIS: False, PUNTAJE: ":.2f", "Depresión (%)": ":.2f",
                "Ansiedad (%)": ":.2f", "Índice de crisis de salud mental": ":.2f"},
    color_continuous_scale="YlGnBu",
    title="Felicidad y salud mental promedio por región"
)
fig.update_layout(geo=dict(showframe=False, showcoastlines=False,
                           projection_type="natural earth"))
fig.write_html(BASE_DIR.parent / "graficos" / "mapa_regiones.html")  # solo HTML, sin kaleido
fig.show()

# ====================================================
# DESEMPEÑO POR REGIÓN (z-scores vs promedio de los países)
# ====================================================

# Variables y si "más alto = mejor" (True) o "más alto = peor" (False)
vars_analisis = {
    "PIB per cápita": True, "Apoyo social": True,
    "Esperanza de vida saludable": True, "Libertad": True,
    "Generosidad": True, "Percepción de corrupción": False,
    "Depresión (%)": False, "Ansiedad (%)": False,
    "Suicidios cada 100k hab.": False, "Brecha de tratamiento (%)": False,
    "Crisis de salud mental juvenil": False, "Índice de crisis de salud mental": False,
    "Calidad del sistema de salud mental": True,   # verificar el sentido en la fuente
}

cols = list(vars_analisis.keys())

# z-score de cada variable respecto a todos los países
z = (df_final[cols] - df_final[cols].mean()) / df_final[cols].std()

# Invertir las que "más alto = peor", así positivo siempre = mejor
for c, mejor_alto in vars_analisis.items():
    if not mejor_alto:
        z[c] = -z[c]

z[REGION] = df_final[REGION]
z_region = z.groupby(REGION)[cols].mean()

# Grilla de subplots
regiones = z_region.index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)

for ax, reg in zip(axes.flat, regiones):
    valores = z_region.loc[reg].sort_values()
    colores = ['#55A868' if v > 0 else '#C44E52' for v in valores]
    ax.barh(valores.index, valores.values, color=colores)
    ax.axvline(0, color='black', linewidth=0.8)
    n = (df_final[REGION] == reg).sum()
    ax.set_title(f"{reg} (n={n})", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

plt.suptitle("Desempeño por región vs promedio de los países (positivo = mejor)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(BASE_DIR.parent / "graficos" / "regiones_desempeno.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

# Variables predictoras.
# NO incluimos "Ranking": sale directo del puntaje, sería hacer trampa.
vars_whr = ["PIB per cápita", "Apoyo social", "Esperanza de vida saludable",
            "Libertad", "Generosidad", "Percepción de corrupción"]
vars_mh = ["Depresión (%)", "Ansiedad (%)", "Suicidios cada 100k hab.",
           "Brecha de tratamiento (%)", "Crisis de salud mental juvenil",
           "Índice de crisis de salud mental", "Calidad del sistema de salud mental"]
todas = vars_whr + vars_mh
 
# Solo países con todos los datos (los mismos para todos los modelos, así se comparan justo)
datos = df_final[[PAIS, PUNTAJE] + todas].dropna().reset_index(drop=True)
y = datos[PUNTAJE]
print(f"\nModelo predictivo: {len(datos)} países")


# ==============================================
# TABLA 1 - REGRESIONES LINEALES
# ==============================================

# Seleccionamos las variables
columnas = [
    PUNTAJE,
    REGION,
    "PIB per cápita",
    "Índice de crisis de salud mental"
]

datos_tabla1 = df_final[columnas].dropna().copy()

# Y es la felicidad que queremos explicar
y_tabla1 = datos_tabla1[PUNTAJE]

# Regresión 1: solo región
X1 = pd.get_dummies(
    datos_tabla1[[REGION]],
    drop_first=True,
    dtype=float
)

# Regresión 2: región + PIB
X2 = X1.copy()
X2["PIB per cápita"] = datos_tabla1["PIB per cápita"]

# Regresión 3: región + PIB + salud mental
X3 = X2.copy()
X3["Índice de crisis de salud mental"] = datos_tabla1[
    "Índice de crisis de salud mental"
]

# Función que calcula el R²
def calcular_r2(X):
    modelo = LinearRegression()
    modelo.fit(X, y_tabla1)
    return modelo.score(X, y_tabla1)

# Calculamos el R² de cada regresión
r2_1 = calcular_r2(X1)
r2_2 = calcular_r2(X2)
r2_3 = calcular_r2(X3)

# Armamos la tabla
tabla_1 = pd.DataFrame({
    "Modelo": ["R1", "R2", "R3"],
    "Variables": [
        "Región",
        "Región + PIB",
        "Región + PIB + salud mental"
    ],
    "R²": [r2_1, r2_2, r2_3]
})

print("\nPRIMERA TABLA DE REGRESIONES")
print(tabla_1.round(3).to_string(index=False))

print("Aporte del PIB:", round(r2_2 - r2_1, 3))
print("Aporte de salud mental:", round(r2_3 - r2_2, 3))


# ----------------------------------------------------
# 1) COMPARAR MODELOS CON VALIDACIÓN CRUZADA
# ----------------------------------------------------
# Con pocos países no conviene un único train/test: se parte en 5 grupos y cada
# grupo se usa una vez como "examen" (el modelo nunca vio esos países al entrenar).
cv = KFold(n_splits=5, shuffle=True, random_state=42)
 
modelos = {
    "Promedio (referencia)": DummyRegressor(strategy="mean"),
    "Regresión lineal": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42),
}
conjuntos = {
    "Solo World Happiness Report": vars_whr,
    "WHR + salud mental": todas,
}
 
filas = []
for nombre_conj, cols_conj in conjuntos.items():
    for nombre_mod, mod in modelos.items():
        res = cross_validate(mod, datos[cols_conj], y, cv=cv,
                             scoring=("r2", "neg_root_mean_squared_error"))
        filas.append({
            "Variables": nombre_conj,
            "Modelo": nombre_mod,
            "R²": res["test_r2"].mean(),
            "Error promedio (RMSE)": -res["test_neg_root_mean_squared_error"].mean(),
        })
 
resultados = pd.DataFrame(filas).round(2)
print("\nResultados (validación cruzada, 5 grupos):")
print(resultados.to_string(index=False))
 
# Gráfico de comparación (R² por modelo y conjunto de variables)
tabla_r2 = resultados.pivot(index="Modelo", columns="Variables", values="R²")
tabla_r2 = tabla_r2.loc[list(modelos.keys())]
fig, ax = plt.subplots(figsize=(8, 5))
tabla_r2.plot(kind="bar", ax=ax, color=['#4C72B0', '#55A868'])
ax.set_title("Qué tan bien predice cada modelo el puntaje de felicidad", fontsize=13, fontweight='bold')
ax.set_ylabel("R² (1 = predice perfecto, 0 = no mejora al promedio)")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=0)
ax.grid(axis="y", alpha=0.3)
ax.legend(title="")
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
plt.tight_layout()
fig.savefig(BASE_DIR.parent / "graficos" / "modelo_comparacion_r2.png", dpi=300, bbox_inches="tight")
plt.close(fig)
 
# ----------------------------------------------------
# 2) QUÉ VARIABLES PESAN MÁS (regresión lineal)
# ----------------------------------------------------
lineal = make_pipeline(StandardScaler(), LinearRegression()).fit(datos[todas], y)
coefs = pd.Series(lineal[-1].coef_, index=todas).sort_values()
print("\nCoeficientes (cambio en el puntaje por cada +1 desviación estándar):")
print(coefs.round(2))
 
colores = ['#55A868' if v in vars_mh else '#4C72B0' for v in coefs.index]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(coefs.index, coefs.values, color=colores)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Qué variables pesan más en el puntaje de felicidad", fontsize=13, fontweight='bold')
ax.set_xlabel("Cambio en el puntaje por cada +1 desviación estándar (azul: WHR, verde: salud mental)")
ax.grid(axis="x", alpha=0.3)
for lado in ("top", "right", "left"):
    ax.spines[lado].set_visible(False)
plt.tight_layout()
fig.savefig(BASE_DIR.parent / "graficos" / "modelo_importancia_variables.png", dpi=300, bbox_inches="tight")
plt.close(fig)
 
# ----------------------------------------------------
# 3) PUNTAJE REAL VS PREDICHO: ¿QUÉ PAÍSES SORPRENDEN?
# ----------------------------------------------------
# Predicciones "honestas": cada país se predice con un modelo que no lo vio al entrenar.
pred = cross_val_predict(make_pipeline(StandardScaler(), LinearRegression()),
                         datos[todas], y, cv=cv)
datos["Puntaje predicho"] = pred
datos["Error"] = datos[PUNTAJE] - datos["Puntaje predicho"]
 
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(datos["Puntaje predicho"], datos[PUNTAJE], color='#4C72B0', alpha=0.75,
           edgecolor='white', linewidth=0.5, s=40)
lim = [min(y.min(), pred.min()) - 0.2, max(y.max(), pred.max()) + 0.2]
ax.plot(lim, lim, color='#D62728', linewidth=2)   # línea de predicción perfecta
for _, fila in datos.reindex(datos["Error"].abs().nlargest(5).index).iterrows():
    ax.annotate(fila[PAIS], (fila["Puntaje predicho"], fila[PUNTAJE]),
                xytext=(6, 6), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
ax.set_title("Puntaje de felicidad real vs predicho por el modelo", fontsize=13, fontweight='bold')
ax.set_xlabel("Puntaje predicho")
ax.set_ylabel("Puntaje real")
ax.grid(alpha=0.2)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
plt.tight_layout()
fig.savefig(BASE_DIR.parent / "graficos" / "modelo_real_vs_predicho.png", dpi=300, bbox_inches="tight")
plt.close(fig)
 
print("\nPaíses MÁS felices de lo que predice el modelo:")
print(datos.nlargest(5, "Error")[[PAIS, PUNTAJE, "Puntaje predicho", "Error"]].round(2).to_string(index=False))
print("\nPaíses MENOS felices de lo que predice el modelo:")
print(datos.nsmallest(5, "Error")[[PAIS, PUNTAJE, "Puntaje predicho", "Error"]].round(2).to_string(index=False))
 
# ----------------------------------------------------
# 4) SIMULACIÓN: "¿QUÉ PASARÍA SI MEJORARA ESTA VARIABLE?"
# ----------------------------------------------------
def simular_mejora(pais, variable, desvios=1.0):
    '''
    Predice cuánto cambiaría el puntaje de un país si la variable subiera
    `desvios` desviaciones estándar (usa la regresión lineal de arriba).
    Es una estimación orientativa: el modelo encuentra asociaciones, no causas.
    '''
    fila = datos.loc[datos[PAIS] == pais, todas].copy()
    if fila.empty:
        print(f"{pais} no está en los datos del modelo.")
        return
    antes = lineal.predict(fila)[0]
    fila[variable] = fila[variable] + desvios * datos[variable].std()
    despues = lineal.predict(fila)[0]
    print(f"{pais}: {desvios:+g} desvío en '{variable}' -> puntaje predicho "
          f"{antes:.2f} -> {despues:.2f} ({despues - antes:+.2f})")
