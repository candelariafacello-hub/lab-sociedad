"""
PROYECTO FINAL — INTRODUCCIÓN A LA CIENCIA DE DATOS
SOCIEDAD: Felicidad y bienestar
Camila Barboza, Belén Etchegoimberry, Candelaria Facello y Clara Gimenez

Pregunta: ¿qué factores explican realmente la felicidad de un país?
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # gráficos interactivos
import plotly.io as pio # configura dnde ver gráficos interactivos
import pycountry_convert as pc # convierte nombre pais a continente
import statsmodels.api as sm # regresión lineal (R², R² ajustado y coeficientes)

# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
GRAF_DIR = BASE_DIR.parent / "graficos"

# 1. DATASET DE FELICIDAD

# 1.1 CARGA Y PRIMERA VISTA DATOS FELICIDAD
# Ruta dataset del World Happiness Report
ruta_csv_happiness = DATA_DIR / "world_happiness_Update_report_2026.csv"

# Verificar que el archivo realmente existe antes de cargarlo
if not ruta_csv_happiness.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv_happiness}")

# Cargar el archivo CSV
df_happiness = pd.read_csv(ruta_csv_happiness)

print(f"--- Archivo cargado exitosamente desde: {ruta_csv_happiness.name} ---")

print("\n\nINFO DEL DATASET DE FELICIDAD ORIGINAL:")
print("Dimensiones (filas, columnas):", df_happiness.shape)
print("\nPrimeras 5 filas:")
print(df_happiness.head())
print("\nTipos de datos y nulos:")
df_happiness.info() # Vemos que no hay nulos
print("\nResumen:")
print(df_happiness.describe().round(2))

# 1.2 LIMPIEZA: países duplicados
print("Filas:", len(df_happiness))
print("Países únicos:", df_happiness["country"].nunique())
print("Filas repetidas por país:", df_happiness["country"].duplicated().sum()) # Problema: Paises duplicados

# Miramos duplicados
duplicados = df_happiness[df_happiness["country"].duplicated(keep=False)]
print(duplicados.sort_values("country").to_string())

# Decisión: promediar las filas repetidas de cada país.
df_happiness_limpio = (
    df_happiness.drop(columns="rank") # El ranking no se promedia
    .groupby("country", as_index=False)
    .mean(numeric_only=True)
)

print("\nHay 17 países repetidos.") 
print("Las dos filas de cada país tienen puntajes distintos.")
print("Decidimos promediarlas para quedarnos con un dato por país")

# 1.3 RENOMBRAR COLUMNAS AL ESPAÑOL
NOMBRES_ES_WH = {
    "country": "País",
    "gdp_per_capita": "PIB per cápita",
    "social_support": "Apoyo social",
    "healthy_life_expectancy": "Esperanza de vida saludable",
    "freedom": "Libertad",
    "generosity": "Generosidad",
    "corruption": "Percepción de corrupción",
    "score": "Puntaje de felicidad",
}
df_happiness_limpio = df_happiness_limpio.rename(columns=NOMBRES_ES_WH)

# Atajos para no repetir textos largos
PAIS = NOMBRES_ES_WH["country"]
PUNTAJE = NOMBRES_ES_WH["score"]
REGION = "Región"

print("\n\nINFO DEL DATASET DE FELICIDAD LIMPIO:")
print("Filas:", len(df_happiness_limpio))
print("Países únicos:", df_happiness_limpio[PAIS].nunique())
print("Duplicados:", df_happiness_limpio[PAIS].duplicated().sum())
df_happiness_limpio.info()

print("\nPrimeras 5 filas:")
print(df_happiness_limpio.head())

# 1.4 CORRELACIONES CON EL PUNTAJE DE FELICIDAD
def grafico_correlacion(df, var_x, archivo, var_y=PUNTAJE, color='#4C72B0', n_atipicos=5):
    '''
    Crea un gráfico de correlación entre var_x y var_y (por defecto el puntaje de felicidad).
    Muestra:
    - una recta de regresión lineal simple (ajuste por mínimos cuadrados)
    - el coeficiente de correlación de Pearson
    - los n_atipicos países más alejados de la recta (mayor residuo en valor
      absoluto), etiquetados con su nombre. Se calcula solo para cada variable.
    '''
    datos = df[[PAIS, var_x, var_y]].dropna() # Quedarse solo con filas completas para x e y
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
    plt.savefig(GRAF_DIR/ f'correlacion_{archivo}.png', dpi=300, bbox_inches="tight")
    plt.close(fig)

# Llamamos función correlaciones con variables de felicidad:
grafico_correlacion(df_happiness_limpio, 'PIB per cápita', 'pib')
grafico_correlacion(df_happiness_limpio, 'Apoyo social', 'apoyo_social')
grafico_correlacion(df_happiness_limpio, 'Esperanza de vida saludable','esperanza_vida')
grafico_correlacion(df_happiness_limpio, 'Libertad', 'libertad')
grafico_correlacion(df_happiness_limpio, 'Generosidad', 'generosidad')
grafico_correlacion(df_happiness_limpio, 'Percepción de corrupción', 'corrupción')

# Tabla de puntaje de correlación de las variables 
variables_whr = ["PIB per cápita", "Apoyo social", "Esperanza de vida saludable",
                 "Libertad", "Generosidad", "Percepción de corrupción"]

print("\n\nCORRELACIONES DE LAS VARIABLES DEL WHR CON EL PUNTAJE DE FELICIDAD:")
print(df_happiness_limpio[variables_whr + [PUNTAJE]].corr()[PUNTAJE].drop(PUNTAJE).round(2))

print("\nLas correlaciones son altas porque estas variables son con las que el WHR construye y explica el puntaje de felicidad.")


# 2. DATASET DE SALUD MENTAL

# ¿Por qué lo sumamos?
# En la sección 1 vimos que las variables del WHR correlacionan fuerte con el puntaje porque son los criterios con los que se calcula.
# Se nos ocurrió sumar el Global Mental Health Crisis Index 2026,
# para ver si la salud mental de un país ayuda a explicar su nivel de felicidad.

# 2.1 CARGA Y PRIMERA VISTA DATOS SALUD MENTAL
ruta_csv_mental = DATA_DIR / "Global_Mental_Health_Crisis_Index_2026.csv"

if not ruta_csv_mental.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv_mental}")

mental = pd.read_csv(ruta_csv_mental)

print("\n\nINFO DEL DATASET DE SALUD MENTAL ORIGINAL:")
print("Dimensiones (filas, columnas):", mental.shape)
print("\nPrimeras 5 filas:")
print(mental.head())
print("\nTipos de datos y nulos:")
mental.info() # No hay valores nulos
print("Países únicos:", mental["country"].nunique())
print("Filas repetidas por país:", mental["country"].duplicated().sum())

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

# Renombrar columnas al español
NOMBRES_ES_SM = {
    "country": "País",
    "depression_pct": "Depresión (%)",
    "anxiety_pct": "Ansiedad (%)",
    "suicide_rate_per100k": "Suicidios cada 100k hab.",
    "treatment_gap_pct": "Brecha de tratamiento (%)",
    "youth_mh_crisis_score": "Crisis de salud mental juvenil",
    "mh_crisis_index": "Índice de crisis de salud mental",
    "mh_system_score": "Calidad del sistema de salud mental",
}
mental_limpio = mental_limpio.rename(columns=NOMBRES_ES_SM)

print("\nDataset de salud mental limpio: conservamos solo variables pertinentes")
print(mental_limpio.head())

# 2.2 UNIÓN DE LOS DOS DATASETS
print("\nCantidad de países en el dataset de felicidad:", df_happiness_limpio[PAIS].nunique())
print("Cantidad de países en el dataset de salud mental:", mental["country"].nunique())  # menos
print("Nota: Al unir los dos datasets se van a perder los países sin registros en el dataset de salud mental")

# Unimos por país. how="inner" conserva solo los países que están en los dos datasets
df_final = pd.merge(df_happiness_limpio, mental_limpio, on=PAIS, how="inner")

print("\n\nINFO DATASET FINAL:")
df_final.info()
print("\nPrimeras 5 filas:")
print(df_final.head())

print("\nCantidad de países antes de unir:", len(df_happiness_limpio))
print("Cantidad de países después de unir:", len(df_final))
print("Duplicados por país:", df_final[PAIS].duplicated().sum()) # sin duplicados

# 2.3 CORRELACIONES CON LAS VARIABLES DE SALUD MENTAL
# Usamos df_final porque es el que tiene las variables de salud mental unidas al puntaje.

# Llamamos la función de correlaciones con las variables de salud mental (en verde)
grafico_correlacion(df_final, 'Depresión (%)', 'depresion', color='#55A868')
grafico_correlacion(df_final, 'Ansiedad (%)', 'ansiedad', color='#55A868')
grafico_correlacion(df_final, 'Suicidios cada 100k hab.', 'suicidios', color='#55A868')
grafico_correlacion(df_final, 'Brecha de tratamiento (%)', 'brecha_tratamiento', color='#55A868')
grafico_correlacion(df_final, 'Crisis de salud mental juvenil', 'crisis_juvenil', color='#55A868')
grafico_correlacion(df_final, 'Índice de crisis de salud mental', 'indice_crisis', color='#55A868')
grafico_correlacion(df_final, 'Calidad del sistema de salud mental', 'sistema_salud_mental', color='#55A868')

# Tabla de puntajes de correlación con las variables de salud mental
variables_salud_mental = [
    "Depresión (%)", "Ansiedad (%)", "Suicidios cada 100k hab.",
    "Brecha de tratamiento (%)", "Crisis de salud mental juvenil",
    "Índice de crisis de salud mental", "Calidad del sistema de salud mental"
]

print("\n\nCORRELACIONES DE LAS VARIABLES DE SALUD MENTAL CON EL PUNTAJE DE FELICIDAD:")
print(df_final[variables_salud_mental + [PUNTAJE]].corr()[PUNTAJE].drop(PUNTAJE).round(2))


# 3. EXPLORACIÓN 

print("\n\nEXPLORACIÓN:")
# 3.1 BARRAS DE RESUMEN: correlación de cada variable con el puntaje de felicidad
def grafico_barras_correlacion(df, variables, color, titulo, archivo, nombre_grupo):
    '''
    Barras horizontales con la correlación de cada variable con el puntaje.
    Imprime la variable más y menos asociada al puntaje (en valor absoluto).
    '''
    correlaciones = df[variables + [PUNTAJE]].corr()[PUNTAJE].drop(PUNTAJE).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(correlaciones.index, correlaciones.values, color=color)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    ax.set_xlabel("Correlación con el puntaje de felicidad")
    ax.axvline(0, linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    plt.tight_layout()
    fig.savefig(GRAF_DIR / f"resumen_correlacion_variables_{archivo}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fuerte = correlaciones.abs().idxmax()
    debil = correlaciones.abs().idxmin()
    print(f"\n→ {nombre_grupo}: la más asociada al puntaje es '{fuerte}' (r = {correlaciones[fuerte]:.2f}) "
          f"y la menos asociada es '{debil}' (r = {correlaciones[debil]:.2f}).")


grafico_barras_correlacion(
    df_happiness_limpio, variables_whr, '#4C72B0',
    "Correlación de variables del World Happiness Report con el puntaje de felicidad",
    "world_happiness_report", "Variables del WHR")

grafico_barras_correlacion(
    df_final, variables_salud_mental, '#55A868',
    "Correlación de variables de salud mental con el puntaje de felicidad",
    "salud_mental", "Variables de salud mental")

print("\nLas variables del WHR tienen correlaciones muy altas con el puntaje (cerca de 0,9 o más), excepto generosidad (aprox  0,25).")
print("Tiene sentido, porque son los criterios con los que se calcula el puntaje.")
print()
print("Las variables de salud mental se comportan distinto, y se separan en tres grupos:")
print("  - Correlación negativa fuerte: índice de crisis, crisis juvenil y brecha de tratamiento (aprox -0,8). A más crisis y más gente sin tratamiento, menos felicidad.")
print("  - Correlación positiva fuerte: calidad del sistema de salud mental (aprox 0,8).")
print("  - Correlación débil o inesperada: ansiedad (casi cero), suicidios (aprox -0,2) y depresión, que da positiva (aprox 0,45)")
print()
print("Lo que parece importar no es cuánta depresión o ansiedad hay, sino cómo responde el país (calidad del sistema y acceso a tratamiento).")
print("Por eso vamos a profundizar en las variables de salud mental.")

# 3.2 PROMEDIOS DE LAS VARIABLES EN LOS 10 PAÍSES MÁS Y MENOS FELICES
variables_todas = variables_whr + variables_salud_mental
top10 = df_final.nlargest(10, PUNTAJE)
bottom10 = df_final.nsmallest(10, PUNTAJE)

comparacion = pd.DataFrame({
    "10 países con mayor felicidad": top10[variables_todas].mean(),
    "10 países con menor felicidad": bottom10[variables_todas].mean(),
})
print("\n\nPROMEDIO DE LAS VARIABLES EN LOS 10 PAÍSES MÁS Y MENOS FELICES:")
print(comparacion.round(2))

# Dos gráficos porque las variables tienen rangos muy distintos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

comparacion.loc[variables_whr].plot(kind="bar", ax=ax1, color=['#E8A33D', '#8C6BAE'])
ax1.set_title("Variables del World Happiness Report", fontsize=12, fontweight='bold')

comparacion.loc[variables_salud_mental].plot(kind="bar", ax=ax2, color=['#E8A33D', '#8C6BAE'])
ax2.set_title("Variables de salud mental", fontsize=12, fontweight='bold')

for ax in (ax1, ax2):
    ax.set_ylabel("Promedio")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")

plt.suptitle("Promedio de variables: 10 países con mayor y menor felicidad", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(GRAF_DIR / "top10_vs_bottom10_felicidad.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Como las variables tienen unidades distintas, medimos la diferencia en desvíos estándar 
# (diferencia entre grupos / desvío de la variable en todos los países)
dif = ((comparacion.iloc[:, 0] - comparacion.iloc[:, 1]) / df_final[variables_todas].std()).abs()
dif = dif.sort_values(ascending=False)
print(f"\n→ Las variables que más separan a los 10 países más felices de los 10 menos felices "
      f"son '{dif.index[0]}' y '{dif.index[1]}'.")

print("Las 6 variables del WHR son más altas en los países más felices.")
print("Las mayores diferencias están en PIB per cápita (1,75 vs 0,6) y en Percepción de corrupción (0,40 vs 0,06). Generosidad es la que menos separa a los grupos.")
print()
print("En salud mental pasa algo llamativo:")
print("  - Depresión y ansiedad son casi iguales en los dos grupos (entre 4% y 6%)")
print("  - Lo que sí separa a los grupos es la respuesta del sistema: la brecha de tratamiento")
print("    pasa de alrededor de 29% a 90% y la calidad del sistema de salud mental de 69 a 7.")
print()
print("Es decir, en los países menos felices casi 9 de cada 10 personas con enfermedad mental no reciben tratamiento.")
print("Una posible explicación es que donde no hay sistema de salud mental la depresión se diagnostica y registra menos, y por eso el porcentaje parece bajo.")

# 3.3 ANÁLISIS POR REGIÓN

print("\n\nANÁLISIS POR REGIÓN:")
pio.renderers.default = "browser"   # abre el mapa en el navegador

# Región (continente) de cada país.
# pycountry_convert necesita los nombres en inglés (como vienen en el CSV).

# Estos países no los reconoce la librería, así que los asignamos a mano:
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
print("\n\nPaíses sin región:", df_final.loc[df_final[REGION].isna(), PAIS].tolist())
df_final = df_final.dropna(subset=[REGION])

# Promedios por región
vars_region = [PUNTAJE, "Depresión (%)", "Ansiedad (%)", "Índice de crisis de salud mental"]
promedios = df_final.groupby(REGION)[vars_region].mean().round(2)
print("\nPromedios por región:")
print(promedios)
print("\nPaíses por región:")
print(df_final[REGION].value_counts())

# Cada país toma los valores promedio de su región
mapa_df = df_final[[PAIS, REGION]].merge(promedios, on=REGION)

# Mapa: color = felicidad promedio de la región
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
fig.write_html(GRAF_DIR / "mapa_regiones.html")
fig.show()

mejor = promedios[PUNTAJE].idxmax()
peor = promedios[PUNTAJE].idxmin()
print(f"\n→ La región más feliz en promedio es {mejor} ({promedios.loc[mejor, PUNTAJE]:.2f}) "
      f"y la menos feliz es {peor} ({promedios.loc[peor, PUNTAJE]:.2f}).")

# 3.4 DESEMPEÑO POR REGIÓN 
# (puntaje z: cuánto se aleja cada región del promedio de los países)

# Variables en las que un valor más alto es PEOR. Se invierten para que positivo siempre signifique "mejor que el promedio".
mas_alto_es_peor = ["Depresión (%)", "Ansiedad (%)", "Suicidios cada 100k hab.",
                    "Brecha de tratamiento (%)", "Crisis de salud mental juvenil",
                    "Índice de crisis de salud mental"]

# z-score de cada variable respecto a todos los países
z = (df_final[variables_todas] - df_final[variables_todas].mean()) / df_final[variables_todas].std()
z[mas_alto_es_peor] = -z[mas_alto_es_peor]
z[REGION] = df_final[REGION]
z_region = z.groupby(REGION)[variables_todas].mean()

# Un gráfico por región
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
fig.savefig(GRAF_DIR / "regiones_desempeno.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n→ Mayor fortaleza y mayor debilidad de cada región:")
for reg in z_region.index:
    print(f"   {reg}: fortaleza = {z_region.loc[reg].idxmax()}; debilidad = {z_region.loc[reg].idxmin()}")

print("\nLas regiones se diferencian mucho entre sí:")
print("  - Europa (n=24) está por encima del promedio en casi todo. Sus mayores fortalezas son calidad del sistema de SM, brecha de tratamiento,")
print("    crisis juvenil e índice de crisis, todas alrededor de +1, incluso por encima del PIB per cápita.")
print("  - África (n=15) es el peor caso: está por debajo del promedio en casi todas las variables,")
print("    con las mayores distancias en esperanza de vida, apoyo social, índice de crisis y PIB.")
print("  - Asia (n=25) queda cerca del promedio en casi todo. Su punto más débil es la calidad del sistema de salud mental y la brecha de tratamiento.")
print("  - América del Sur (n=7) queda algo por debajo del promedio, con la ansiedad como mayor debilidad.")
print("  - América del Norte (n=4) y Oceanía (n=2) tienen muy buen desempeño general, pero son pocos países, así que sus promedios no son tan confiables.")
print()
print("Notamos nuevamente que las regiones más felices (Europa, América del Norte y Oceanía) tienen más depresión que el promedio y África y Asia tienen menos.")
print()
print("Como el desempeño cambia tanto de una región a otra, en los modelos vamos a sumar la región.")


# 4. MODELOS

# Armamos modelos de regresión lineal que estiman el puntaje de felicidad de un país
# a partir de otras variables, con una fórmula del tipo:  Y = a + b1·x1 + b2·x2 + ...

# Para comparar los modelos usamos el R²:
#   R² = 1 - (error del modelo / variación total del puntaje)
#   R² = 0 -> el modelo no mejora a predecir el puntaje de felicidad
#   R² = 1 -> el modelo predice perfecto

# Problema: el R² sube siempre que agregamos una variable, aunque no aporte nada.
# ==> Por eso también miramos el R² AJUSTADO, que penaliza las variables que no suman

# Un R² alto muestra que las variables están asociadas a la felicidad, no que la causen.
 
print("\n\nMODELOS DE REGRESIÓN LINEAL:")
print("Países usados en todos los modelos:", len(df_final))
 
# Atajos para no repetir textos largos
PIB = "PIB per cápita"
BRECHA = "Brecha de tratamiento (%)"
SISTEMA = "Calidad del sistema de salud mental"
JUVENIL = "Crisis de salud mental juvenil"
DEPRESION = "Depresión (%)"
INDICE = "Índice de crisis de salud mental"
 
 
def ajustar_modelo(df, variables, usar_region=False):
    '''
    Ajusta una regresión lineal del puntaje de felicidad con las variables indicadas.
    Si usar_region=True suma la región como variables 0/1 (vale 1 si el país es de esa
    región y 0 si no). Europa queda como región base: el coeficiente de cada región
    es la diferencia de felicidad respecto a Europa.
    '''
    X = df[variables].copy()
    if usar_region:
        regiones_0_1 = pd.get_dummies(df[REGION], dtype=float).drop(columns="Europa")
        X = X.join(regiones_0_1)
    X = sm.add_constant(X)  # agrega la constante "a" de la fórmula
    return sm.OLS(df[PUNTAJE], X).fit()
 
 
def tabla_r2(df, modelos):
    '''
    Recibe un diccionario {nombre del modelo: (variables, usar_region)},
    ajusta cada modelo y devuelve una tabla con su R² y R² ajustado.
    '''
    filas = []
    for nombre, (variables, usar_region) in modelos.items():
        modelo = ajustar_modelo(df, variables, usar_region)
        filas.append({
            "Modelo": nombre,
            "Nº de variables": len(modelo.params) - 1,  # sin contar la constante
            "R²": modelo.rsquared,
            "R² ajustado": modelo.rsquared_adj,
        })
    return pd.DataFrame(filas).set_index("Modelo").round(3)
 
 
# 4.1 MODELO CON LAS VARIABLES DEL WORLD HAPPINESS REPORT
# Primero comprobamos lo que esperábamos de la sección 1: como el puntaje se construye
# con estas variables, el R² debería dar casi 1.
modelos_whr = {
    "Solo apoyo social": (["Apoyo social"], False),
    "Las 6 variables del WHR": (variables_whr, False),
}
tabla_whr = tabla_r2(df_final, modelos_whr)
 
print("\n\nR² CON LAS VARIABLES DEL WHR:")
print(tabla_whr.to_string())
 
print(f"\n→ Con solo apoyo social el modelo ya explica el {tabla_whr.loc['Solo apoyo social', 'R²']:.0%} "
      f"de las diferencias de felicidad, y con las 6 variables el {tabla_whr.loc['Las 6 variables del WHR', 'R²']:.0%}.")
print("Esto no es un descubrimiento: estas variables son con las que el WHR construye el puntaje.")
print("Por eso dejamos de lado estas variables y nos enfocamos en las de salud mental,")
print("que vienen de otro dataset y sí pueden aportar información nueva.")
 
 
# 4.2 ELECCIÓN DE LAS VARIABLES DE SALUD MENTAL
# Usamos lo que vimos en la exploración (sección 3):
#   - Brecha de tratamiento (r aprox -0,81): fuerte y fácil de interpretar (gente sin tratamiento).
#   - Calidad del sistema de salud mental (r aprox 0,78): fuerte, mide cómo responde el país.
#   - Crisis de salud mental juvenil (r aprox -0,84): fuerte.
#   - Depresión (r aprox 0,46): da positiva, al revés de lo esperado. La probamos para ver
#     si aporta algo cuando ya están las otras.
# Quedan afuera:
#   - Ansiedad (r aprox -0,01): no tiene relación lineal con la felicidad.
#   - Suicidios (r aprox -0,18): relación débil y arrastrada por 2 países con valores extremos.
#   - Índice de crisis (r aprox -0,86): es un índice compuesto (0-100) armado a partir de
#     otros indicadores de salud mental. Combinarlo con ellos sería repetir información,
#     así que solo lo usamos solo, como referencia.
 
candidatas = [BRECHA, SISTEMA, JUVENIL, DEPRESION]
 
# Antes de combinarlas miramos si están muy correlacionadas ENTRE SÍ.
# Si dos variables dan casi la misma información, usarlas juntas casi no mejora el modelo.
corr_candidatas = df_final[candidatas + [INDICE]].corr().round(2)
print("\n\nCORRELACIÓN ENTRE LAS VARIABLES DE SALUD MENTAL:")
print(corr_candidatas.to_string())
 
print("\n→ Pares de variables muy correlacionadas entre sí (|r| > 0,8):")
pares_altos = 0
for i, var1 in enumerate(corr_candidatas.columns):
    for var2 in corr_candidatas.columns[i + 1:]:
        r = corr_candidatas.loc[var1, var2]
        if abs(r) > 0.8:
            print(f"   {var1} y {var2}: r = {r:.2f}")
            pares_altos += 1
if pares_altos == 0:
    print("   Ninguno.")
 
 
# 4.3 TABLA DE R² CON VARIABLES DE SALUD MENTAL
# Probamos cada variable sola y después combinaciones, para ver cuáles suman.
modelos_sm = {
    "Índice de crisis (referencia)": ([INDICE], False),
    "Brecha de tratamiento": ([BRECHA], False),
    "Calidad del sistema": ([SISTEMA], False),
    "Crisis juvenil": ([JUVENIL], False),
    "Brecha + Sistema": ([BRECHA, SISTEMA], False),
    "Brecha + Crisis juvenil": ([BRECHA, JUVENIL], False),
    "Brecha + Sistema + Crisis juvenil": ([BRECHA, SISTEMA, JUVENIL], False),
    "Brecha + Sistema + Crisis juvenil + Depresión": ([BRECHA, SISTEMA, JUVENIL, DEPRESION], False),
}
tabla_sm = tabla_r2(df_final, modelos_sm)
 
print("\n\nR² CON VARIABLES DE SALUD MENTAL:")
print(tabla_sm.to_string())
 
# Elegimos la combinación con mayor R² ajustado.
# Dejamos afuera el índice de crisis: no sabemos cómo está calculado,
# así que no podemos interpretar su fórmula.
tabla_sm_componentes = tabla_sm.drop(index="Índice de crisis (referencia)")
mejor_sm = tabla_sm_componentes["R² ajustado"].idxmax()
vars_sm_final = modelos_sm[mejor_sm][0]
 
print(f"\n→ La mejor combinación es '{mejor_sm}' "
      f"(R² ajustado = {tabla_sm.loc[mejor_sm, 'R² ajustado']:.2f}).")
print(f"→ Como referencia, el índice de crisis solo da un R² ajustado de "
      f"{tabla_sm.loc['Índice de crisis (referencia)', 'R² ajustado']:.2f}.")
print("Sin usar ninguna variable del WHR, la salud mental ya explica una parte importante")
print("de las diferencias de felicidad entre países.")
 
 
# 4.4 SUMAMOS LA REGIÓN Y EL PIB COMO CONTROL
# Región: en la sección 3 vimos que el desempeño cambia mucho de una región a otra.
#   Advertencia: Oceanía (2 países) y América del Norte (4) tienen muy pocos países,
#   así que sus coeficientes son poco confiables.
# PIB per cápita: no lo usamos para explicar la felicidad (es una variable del WHR), sino como
#   CONTROL. Los países ricos suelen tener mejores sistemas de salud mental, así que queremos
#   saber si la salud mental aporta algo MÁS ALLÁ de la riqueza del país.
modelos_control = {
    "Solo PIB": ([PIB], False),
    "Solo región": ([], True),
    "Salud mental": (vars_sm_final, False),
    "Salud mental + región": (vars_sm_final, True),
    "Salud mental + PIB": (vars_sm_final + [PIB], False),
    "Salud mental + PIB + región": (vars_sm_final + [PIB], True),
}
tabla_control = tabla_r2(df_final, modelos_control)
 
print("\n\nR² SUMANDO REGIÓN Y PIB (CONTROL):")
print(tabla_control.to_string())
 
aporte_region = tabla_control.loc["Salud mental + región", "R² ajustado"] - tabla_control.loc["Salud mental", "R² ajustado"]
aporte_sm_sobre_pib = tabla_control.loc["Salud mental + PIB", "R² ajustado"] - tabla_control.loc["Solo PIB", "R² ajustado"]
 
print(f"\n→ Sumar la región a la salud mental cambia el R² ajustado en {aporte_region:+.3f}.")
print(f"→ Sumar la salud mental a un modelo que ya tiene el PIB cambia el R² ajustado en {aporte_sm_sobre_pib:+.3f}.")
if aporte_sm_sobre_pib > 0.02:
    print("La salud mental aporta información que el PIB no captura: no es solo un reflejo de la riqueza del país.")
else:
    print("La salud mental casi no aporta una vez que está el PIB: en buena parte refleja la riqueza del país.")
 
 
# 4.5 MODELO FINAL: SALUD MENTAL + REGIÓN
# Es el modelo hacia el que guió el análisis: variables de salud mental elegidas en 4.3
# más la región. No incluimos el PIB porque es una variable del WHR y queremos que el
# modelo se enfoque en la salud mental.
modelo_final = ajustar_modelo(df_final, vars_sm_final, usar_region=True)
 
coeficientes = pd.DataFrame({
    "Coeficiente": modelo_final.params,
    "p-valor": modelo_final.pvalues,
}).round(3)
print("\n\nCOEFICIENTES DEL MODELO FINAL:")
print(coeficientes.to_string())
print("Cada coeficiente indica cuánto cambia el puntaje predicho si esa variable sube 1 unidad")
print("y las demás quedan iguales. En las regiones: diferencia de felicidad respecto a Europa.")
print("p-valor < 0,05 -> hay evidencia de que la variable está asociada a la felicidad.")
 
# Fórmula de la regresión: Y = a + b1·x1 + b2·x2 + ...
NOMBRES_CORTOS = {
    BRECHA: "Brecha", SISTEMA: "Sistema", JUVENIL: "Crisis juvenil",
    DEPRESION: "Depresión", PIB: "PIB",
}
 
def texto_formula(modelo, terminos_por_linea=3):
    '''Arma la fórmula de la regresión como texto, cortando en varias líneas.'''
    terminos = []
    for nombre, b in modelo.params.drop("const").items():
        signo = "+" if b >= 0 else "−"
        terminos.append(f"{signo} {abs(b):.3f}·{NOMBRES_CORTOS.get(nombre, nombre)}")
    lineas = [f"Felicidad = {modelo.params['const']:.2f}"]
    for i in range(0, len(terminos), terminos_por_linea):
        lineas.append("   " + " ".join(terminos[i:i + terminos_por_linea]))
    return "\n".join(lineas)
 
formula = texto_formula(modelo_final)
print("\nFÓRMULA DEL MODELO FINAL:")
print(formula)
print("(Región: vale 1 si el país es de esa región y 0 si no. Europa es la base.)")
 
# Gráfico: puntaje real vs predicho
# Con varias variables no hay una sola recta para dibujar, así que comparamos lo que
# predice el modelo con el puntaje real. Si el modelo fuera perfecto, todos los puntos
# estarían sobre la línea roja.
datos_modelo = df_final[[PAIS, REGION, PUNTAJE]].copy()
datos_modelo["Puntaje predicho"] = modelo_final.fittedvalues
datos_modelo["Error"] = datos_modelo[PUNTAJE] - datos_modelo["Puntaje predicho"]
 
fig, ax = plt.subplots(figsize=(9, 7))
 
for reg, grupo in datos_modelo.groupby(REGION):
    ax.scatter(grupo["Puntaje predicho"], grupo[PUNTAJE], alpha=0.8,
               edgecolor='white', linewidth=0.5, s=50, label=f"{reg} (n={len(grupo)})")
 
lim = [datos_modelo[[PUNTAJE, "Puntaje predicho"]].min().min() - 0.2,
       datos_modelo[[PUNTAJE, "Puntaje predicho"]].max().max() + 0.2]
ax.plot(lim, lim, color='#D62728', linewidth=2, label="Predicción perfecta")
 
# Etiquetamos los 5 países más alejados de lo que predice el modelo
for _, fila in datos_modelo.loc[datos_modelo["Error"].abs().nlargest(5).index].iterrows():
    ax.annotate(fila[PAIS], (fila["Puntaje predicho"], fila[PUNTAJE]),
                xytext=(6, 6), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
 
ax.set_title(f"Modelo final (salud mental + región): puntaje real vs predicho\n"
             f"R² = {modelo_final.rsquared:.2f}  |  R² ajustado = {modelo_final.rsquared_adj:.2f}  |  "
             f"{int(modelo_final.nobs)} países", fontsize=12, fontweight='bold')
ax.set_xlabel("Puntaje predicho por el modelo")
ax.set_ylabel("Puntaje real de felicidad")
ax.grid(alpha=0.2)
ax.legend(fontsize=8, loc="upper left")
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
 
plt.tight_layout()
# Fórmula debajo del gráfico
fig.text(0.5, -0.01, formula + "\n(Región: 1 si el país es de esa región, 0 si no. Base: Europa)",
         ha="center", va="top", fontsize=9, family="monospace")
fig.savefig(GRAF_DIR / "modelo_final_real_vs_predicho.png", dpi=300, bbox_inches="tight")
plt.close(fig)
 
print("\nPaíses MÁS felices de lo que predice el modelo:")
print(datos_modelo.nlargest(3, "Error")[[PAIS, PUNTAJE, "Puntaje predicho", "Error"]].round(2).to_string(index=False))
print("\nPaíses MENOS felices de lo que predice el modelo:")
print(datos_modelo.nsmallest(3, "Error")[[PAIS, PUNTAJE, "Puntaje predicho", "Error"]].round(2).to_string(index=False))
 
print(f"\n→ El modelo final explica el {modelo_final.rsquared:.0%} de las diferencias de felicidad "
      f"entre países usando solo salud mental y región, sin variables del WHR.")

