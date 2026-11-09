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
from sklearn.linear_model import LinearRegression # regresión lineal para los modelos (R² y fórmula)

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

# Problema: el R² sube siempre que agregamos una variable
# ==> Por eso también miramos el R² AJUSTADO, que penaliza las variables que no suman y sube solo si las variabels nuevas aportan

# Un R² alto muestra que las variables están asociadas a la felicidad, no que la causen.
 
print("\n\nMODELOS DE REGRESIÓN LINEAL:")
 
columnas_para_modelos = [PAIS, PUNTAJE, REGION] + variables_whr + variables_salud_mental
datos_modelos = df_final[columnas_para_modelos].copy() # tabla a parte para los modelos
 
y = datos_modelos[PUNTAJE] # Y = variable que queremos explicar
# la difinimos arriba porque todos los modelos explican la misma "y"
 
def r2_ajustado(r2, n, p):
    '''
    (n = cantidad de países, p = cantidad de variables)
    Calcula el R² ajustado: corrige el R² según cuántas variables usa el modelo.
    Solo sube si la variable nueva realmente aporta. 
    '''
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)
# en la función crear_modelo llamo a r2_ajustado y le paso:
# - r2
# - len(y): la cantidad de países (n), porque y tiene un valor por país
# - X.shape[1]: la cantidad de columnas de X (variables del modelo p)
 
def crear_modelo(X):
    '''
    Arma una regresión lineal que explica el puntaje de felicidad con las variables X.
    Devuelve el modelo (con su fórmula), su R² y su R² ajustado.
    '''
    modelo = LinearRegression() # crea un modelo de regresión
    modelo.fit(X, y) # incorpora las variables X ingresadas al modelo
    r2 = modelo.score(X, y) # calcula R²
    return modelo, r2, r2_ajustado(r2, len(y), X.shape[1])

 
# 4.1 MODELO CON LAS VARIABLES DEL WORLD HAPPINESS REPORT
# Primero comprobamos lo que esperábamos en la sección 1, como el puntaje se construye con las variables del WHR, el R² debería dar casi 1.

# Probamos primero crear un modelo solo con apoyo social ya que era la variable del WHR con mayor correlación (r = 0,98)
modelo_apoyo, r2_apoyo, r2a_apoyo = crear_modelo(datos_modelos[["Apoyo social"]])

# Creamos también un modelo con todas las variables del WHR
modelo_whr, r2_whr, r2a_whr = crear_modelo(datos_modelos[variables_whr])
 
print("\n\nR² CON LAS VARIABLES DEL WHR:")
print(f"Solo apoyo social:         R² = {r2_apoyo:.3f} | R² ajustado = {r2a_apoyo:.3f}")
print(f"Las 6 variables del WHR:   R² = {r2_whr:.3f} | R² ajustado = {r2a_whr:.3f}")
 
print(f"\n→ Con solo apoyo social el modelo ya explica el {r2_apoyo:.0%} de las diferencias de felicidad, y con las 6 variables el {r2_whr:.0%}.")
print("Esto se da ya que estas son las variables con las que el WHR construye el puntaje.")
print("Por eso dejamos de lado estas variables y nos enfocamos en las de salud mental.")
 
 
# 4.2 TABLA 1: REGIÓN, PIB Y SALUD MENTAL
# Calcula R² con distintas combinaciones de variables:
#   Modelo 1: Región (en la sección 3 vimos que el desempeño cambia mucho entre regiones)
#   Modelo 2: Región + PIB per cápita (como variable de control)
#   Modelo 3: Región + PIB + salud mental. 
#             Usamos el índice de crisis de salud mental porque es la variable de SM con mayor correlación con la felicidad (r = -0,86)
#             y porque es un índice compuesto que resume toda la salud mental en una sola variable.

# Si el R² ajustado sube del Modelo 2 al Modelo 3, la salud mental aporta información nueva.

# Primero tenemos que pasar la región a numeros ya que la regresión solo trabaja con números y la región es texto.
regiones_0_1 = pd.get_dummies(datos_modelos[REGION], dtype=float).drop(columns="Europa")
# Creamos una columna por región, que vale 1 si el país es de esa región y 0 si no.
# Europa queda como región base (es la de más países)

# Armamos las variables (X) de cada modelo
X_modelo1 = regiones_0_1.copy()

X_modelo2 = X_modelo1.copy()
X_modelo2["PIB per cápita"] = datos_modelos["PIB per cápita"] # sumo al 1 la columna de PBI

X_modelo3 = X_modelo2.copy()
X_modelo3["Índice de crisis de salud mental"] = datos_modelos["Índice de crisis de salud mental"] # sumo al 2 indice de crisis de SM

# Creamos los 3 modelos y guardamos su R² y R² ajustado
modelo1, r2_1, r2a_1 = crear_modelo(X_modelo1)
modelo2, r2_2, r2a_2 = crear_modelo(X_modelo2)
modelo3, r2_3, r2a_3 = crear_modelo(X_modelo3)

# Calculamos cuánto aporta cada bloque nuevo (con el R² ajustado)
aporte_pib = r2a_2 - r2a_1   # lo que suma el PIB a la región
aporte_sm = r2a_3 - r2a_2    # lo que suma la salud mental a la región y el PIB

# Armamos la tabla con los datos calculados
tabla_1 = pd.DataFrame({
    "Modelo": ["Modelo 1", "Modelo 2", "Modelo 3"],
    "Variables": ["Región", "Región + PIB", "Región + PIB + salud mental"],
    "R²": [r2_1, r2_2, r2_3],
    "R² ajustado": [r2a_1, r2a_2, r2a_3],
    "Aporte adicional": [None, aporte_pib, aporte_sm], 
})

print("\n\nTABLA 1: REGIÓN, PIB Y SALUD MENTAL")
print(tabla_1.round(3).to_string(index=False)) # muestro tabla con numeros redondeados y entera (sin ...)

print(f"\n→ Aporte del PIB sobre la región: {aporte_pib:+.3f} de R² ajustado.")
print(f"→ Aporte de la salud mental sobre la región y el PIB: {aporte_sm:+.3f} de R² ajustado.")
if aporte_sm > 0.02:
    print("La salud mental aporta información que la región y el PIB no capturan.")
else:
    print("La salud mental casi no aporta una vez que están la región y el PIB.")
 
 
# 4.3 ELECCIÓN DE LAS VARIABLES DE SALUD MENTAL
# Ahora abrimos el índice de crisis y miramos sus indicadores por separado.
# Usamos lo que vimos en la exploración (sección 3):
#   - Brecha de tratamiento (r aprox -0,81): fuerte y fácil de interpretar (gente sin tratamiento).
#   - Calidad del sistema de salud mental (r aprox 0,78): fuerte, mide cómo responde el país.
#   - Crisis de salud mental juvenil (r aprox -0,84): fuerte.
#   - Depresión (r aprox 0,46): da positiva, al revés de lo esperado. La probamos para ver
#     si aporta algo cuando ya están las otras.
# Quedan afuera:
#   - Ansiedad (r aprox -0,01): no tiene relación lineal con la felicidad.
#   - Suicidios (r aprox -0,18): relación débil y arrastrada por 2 países con valores extremos.
#   - Índice de crisis: es un índice compuesto que ya resume los demás indicadores.
#     Combinarlo con ellos sería repetir información, así que queda solo como referencia.
variables_salud = [
    "Brecha de tratamiento (%)",
    "Calidad del sistema de salud mental",
    "Crisis de salud mental juvenil",
    "Depresión (%)",
]
 
# Antes de combinarlas miramos si están muy correlacionadas ENTRE SÍ.
# Si dos variables dan casi la misma información, usarlas juntas casi no mejora el modelo.
corr_salud = datos_modelos[variables_salud + ["Índice de crisis de salud mental"]].corr().round(2)
print("\n\nCORRELACIÓN ENTRE LAS VARIABLES DE SALUD MENTAL:")
print(corr_salud.to_string())
 
print("\n→ Pares de variables muy correlacionadas entre sí (|r| > 0,8):")
pares_altos = 0
for i, var1 in enumerate(corr_salud.columns):
    for var2 in corr_salud.columns[i + 1:]:
        r = corr_salud.loc[var1, var2]
        if abs(r) > 0.8:
            print(f"   {var1} y {var2}: r = {r:.2f}")
            pares_altos += 1
if pares_altos == 0:
    print("   Ninguno.")
 
 
# 4.4 TABLA 2: REGRESIONES CON VARIABLES DE SALUD MENTAL
# Primero vemos cuánto explica cada variable SOLA.
r2_solas = {}
for var in variables_salud + ["Índice de crisis de salud mental"]:
    _, r2, _ = crear_modelo(datos_modelos[[var]])
    r2_solas[var] = r2
 
print("\n\nR² DE CADA VARIABLE DE SALUD MENTAL POR SEPARADO:")
print(pd.Series(r2_solas, name="R²").sort_values(ascending=False).round(3).to_string())
 
# Después agregamos una variable más en cada regresión, empezando por la que más explica
# sola. El orden importa: la variable que entra primero se lleva el crédito de la
# información que comparte con las demás. Así, el "aporte adicional" mide lo que cada
# variable suma DE NUEVO.
orden_salud = sorted(variables_salud, key=lambda v: r2_solas[v], reverse=True)
 
resultados_tabla2 = []
for i in range(1, len(orden_salud) + 1):
    variables_modelo = orden_salud[:i]
    _, r2, r2a = crear_modelo(datos_modelos[variables_modelo])
    resultados_tabla2.append({
        "Modelo": f"R{i}",
        "Variables": " + ".join(variables_modelo),
        "R²": r2,
        "R² ajustado": r2a,
    })
 
tabla_2 = pd.DataFrame(resultados_tabla2)
tabla_2["Aporte adicional"] = tabla_2["R² ajustado"].diff()
 
print("\n\nTABLA 2: REGRESIONES CON VARIABLES DE SALUD MENTAL")
print(tabla_2.round(3).to_string(index=False))
 
# Elegimos el modelo con mayor R² ajustado
mejor = tabla_2["R² ajustado"].idxmax()
vars_salud_final = orden_salud[:mejor + 1]
print(f"\n→ El mejor modelo es {tabla_2.loc[mejor, 'Modelo']} ({tabla_2.loc[mejor, 'Variables']}), "
      f"con R² ajustado = {tabla_2.loc[mejor, 'R² ajustado']:.3f}.")
print(f"→ Como referencia, el índice de crisis solo tiene R² = {r2_solas['Índice de crisis de salud mental']:.3f}.")
print("Las variables que agregamos después casi no suman porque repiten información")
print("de las que ya estaban (ver la correlación entre ellas en 4.3).")
 
 
# 4.5 MODELO FINAL: SALUD MENTAL + REGIÓN
# Es el modelo hacia el que guió el análisis: las variables de salud mental elegidas en 4.4
# más la región. No incluimos el PIB porque es una variable del WHR y queremos que el
# modelo se enfoque en la salud mental.
X_final = datos_modelos[vars_salud_final].join(regiones_0_1)
modelo_final, r2_final, r2a_final = crear_modelo(X_final)
 
print("\n\nMODELO FINAL (SALUD MENTAL + REGIÓN):")
print(f"R² = {r2_final:.3f} | R² ajustado = {r2a_final:.3f} | {len(y)} países")
print(f"Sin región: R² ajustado = {tabla_2.loc[mejor, 'R² ajustado']:.3f} -> "
      f"la región aporta {r2a_final - tabla_2.loc[mejor, 'R² ajustado']:+.3f}.")
 
coeficientes = pd.Series(modelo_final.coef_, index=X_final.columns, name="Coeficiente")
print("\nCoeficientes:")
print(coeficientes.round(3).to_string())
print("Cada coeficiente indica cuánto cambia el puntaje predicho si esa variable sube 1 unidad")
print("y las demás quedan iguales. En las regiones: diferencia de felicidad respecto a Europa.")
 
# Fórmula de la regresión: Y = a + b1·x1 + b2·x2 + ...
def texto_formula(modelo, columnas_X, terminos_por_linea=2):
    '''Arma la fórmula de la regresión como texto, cortando en varias líneas.'''
    terminos = []
    for nombre, b in zip(columnas_X, modelo.coef_):
        signo = "+" if b >= 0 else "−"
        terminos.append(f"{signo} {abs(b):.3f}·{nombre}")
    lineas = [f"Felicidad = {modelo.intercept_:.2f}"]
    for i in range(0, len(terminos), terminos_por_linea):
        lineas.append("   " + " ".join(terminos[i:i + terminos_por_linea]))
    return "\n".join(lineas)
 
formula = texto_formula(modelo_final, X_final.columns)
print("\nFÓRMULA DEL MODELO FINAL:")
print(formula)
print("(Región: vale 1 si el país es de esa región y 0 si no. Europa es la base.)")
 
# Gráfico: puntaje real vs predicho
# Con varias variables no hay una sola recta para dibujar, así que comparamos lo que
# predice el modelo con el puntaje real. Si el modelo fuera perfecto, todos los puntos
# estarían sobre la línea roja.
datos_modelos["Puntaje predicho"] = modelo_final.predict(X_final)
datos_modelos["Error"] = datos_modelos[PUNTAJE] - datos_modelos["Puntaje predicho"]
 
fig, ax = plt.subplots(figsize=(9, 7))
 
for reg, grupo in datos_modelos.groupby(REGION):
    ax.scatter(grupo["Puntaje predicho"], grupo[PUNTAJE], alpha=0.8,
               edgecolor='white', linewidth=0.5, s=50, label=f"{reg} (n={len(grupo)})")
 
lim = [datos_modelos[[PUNTAJE, "Puntaje predicho"]].min().min() - 0.2,
       datos_modelos[[PUNTAJE, "Puntaje predicho"]].max().max() + 0.2]
ax.plot(lim, lim, color='#D62728', linewidth=2, label="Predicción perfecta")
 
# Etiquetamos los 5 países más alejados de lo que predice el modelo
for _, fila in datos_modelos.loc[datos_modelos["Error"].abs().nlargest(5).index].iterrows():
    ax.annotate(fila[PAIS], (fila["Puntaje predicho"], fila[PUNTAJE]),
                xytext=(6, 6), textcoords='offset points', fontsize=8,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
 
ax.set_title(f"Modelo final (salud mental + región): puntaje real vs predicho\n"
             f"R² = {r2_final:.2f}  |  R² ajustado = {r2a_final:.2f}  |  {len(y)} países",
             fontsize=12, fontweight='bold')
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
print(datos_modelos.nlargest(3, "Error")[[PAIS, PUNTAJE, "Puntaje predicho", "Error"]].round(2).to_string(index=False))
print("\nPaíses MENOS felices de lo que predice el modelo:")
print(datos_modelos.nsmallest(3, "Error")[[PAIS, PUNTAJE, "Puntaje predicho", "Error"]].round(2).to_string(index=False))
 
print(f"\n→ El modelo final explica el {r2_final:.0%} de las diferencias de felicidad entre países")
print("  usando solo salud mental y región, sin variables del WHR.")
 