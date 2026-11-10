Readme · MD
# Felicidad y bienestar: ¿qué factores explican realmente la felicidad de un país?
 
Introducción a Ciencia de Datos – Candelaria Facello, Belen Etchegoimberry, Camila Barboza y Clara Gimenez.
 
## Descripción del problema
 
Existe una creencia extendida de que el ingreso económico es el principal determinante de la felicidad de una población. Sin embargo, el *World Happiness Report* mide la felicidad como una combinación de varios factores: ingreso per cápita, apoyo social, expectativa de vida saludable, libertad para tomar decisiones de vida, generosidad y percepción de corrupción.

Como esas seis variables son justamente los criterios con los que el WHR construye y explica el puntaje, es esperable que correlacionen fuerte con él. Por eso sumamos una fuente externa, el *Global Mental Health Crisis Index 2026*, para ver si la salud mental de un país aporta información nueva para explicar su nivel de felicidad.
 
**Problema a resolver:** Determinar que factores explican la felicidad de un país, y si la salud mental ayuda a explicarla más allá de las variables del WHR. 
 
## Objetivos
 
### Objetivo general
 
Analizar los determinantes de la felicidad a nivel país en 2026, evaluando el peso relativo del ingreso frente a otras variables del World Happiness Report. Integrar los datos del Global Mental Health Crisis Index 2026 para sumar el peso de variables relacionadas a la salud mental.
 
### Objetivos específicos
 
1. Cuantificar la correlación entre las distintas variables (del WHR y de salud mental) y el puntaje de felicidad.
2. Comparar el perfil completo de las variables entre los 10 países con mayor y menor puntaje de felicidad, para identificar en qué dimensiones se diferencian más.
3. Identificar países atípicos: aquellos que se alejan de la tendencia general (por ejemplo, `gdp_per_capita` alto y `score` relativamente bajo, o al revés) y explorar qué otra variable explica esa desviación.
4. Comparar el desempeño de las distintas regiones (continentes) en felicidad y salud mental.
5. Construir un modelo de regresión lineal que estime el puntaje de felicidad a partir de variables de salud mental y región, sin usar las variables del WHR, y medir cuánto explica.
   
## Datasets
 
**World Happiness Update Report 2026** — Kaggle: [kaggle.com/datasets/jummanalnahian/world-happiness-update-report-2026](https://www.kaggle.com/datasets/jummanalnahian/world-happiness-update-report-2026)
 
Corte más reciente del World Happiness Report, con 9 columnas por país: `rank`, `country`, `gdp_per_capita`, `social_support`, `healthy_life_expectancy`, `freedom`, `generosity`, `corruption` y `score`.

**Global Mental Health Crisis Index 2026** — Kaggle: [kaggle.com/datasets/alitaqishah/global-mental-health-crisis-index-2026](https://www.kaggle.com/datasets/alitaqishah/global-mental-health-crisis-index-2026)

Archivo CSV con indicadores de salud mental por país tales como: `country`, `depression_pct`, `anxiety_pct`, `suicide_rate_per100k`, `treatment_gap_pct`, `youth_mh_crisis_score`, `mh_crisis_index`, `mh_system_score`.

## Estructura del repositorio
 
```
lab-sociedad/
├── data/
│   ├── Global_Mental_Health_Crisis_Index_2026.csv
│   └── world_happiness_Update_report_2026.csv
├── graficos/                  # gráficos generados por el script
│   ├── correlacion_*.png              # dispersión de cada variable vs puntaje
│   ├── resumen_correlacion_*.png      # barras de correlación (WHR y salud mental)
│   ├── top10_vs_bottom10_felicidad.png
│   ├── regiones_desempeno.png
│   ├── mapa_regiones.html             # mapa interactivo por región
│   └── modelo_final_real_vs_predicho.png   # regresión lineal
├── scripts/
│   └── Proyecto_felicidad.py
├── Proyecto Felicidad y bienestar.pdf.    # presentación final
├── decisiones.md
└── README.md
```

## Cómo ejecutar
 
1. Instalar las dependencias:
```
pip install pandas numpy matplotlib plotly pycountry-convert statsmodels
```
2. Verificar que los dos CSV estén en `data/` y que exista la carpeta `graficos/`.
3. Desde la carpeta `lab-sociedad`, correr el script:
```
python scripts/Proyecto_felicidad.py
```
Los resultados se imprimen en consola, los gráficos se guardan en `graficos/` y el mapa por región se abre en el navegador.
