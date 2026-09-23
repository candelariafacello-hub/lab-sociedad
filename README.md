Readme · MD
# Felicidad y bienestar: ¿qué factores explican realmente la felicidad de un país?
 
Introducción a Ciencia de Datos – Candelaria Facello, Belen Etchegoimberry, Camila Barboza y Clara Gimenez.
 
## Descripción del problema
 
Existe una creencia extendida de que el ingreso económico es el principal determinante de la felicidad de una población. Sin embargo, el *World Happiness Report* mide la felicidad como una combinación de varios factores: ingreso per cápita, apoyo social, expectativa de vida saludable, libertad para tomar decisiones de vida, generosidad y percepción de corrupción.
 
**Problema a resolver:** Determinar que factores explican la felicidad de un país. 
 
## Objetivos
 
### Objetivo general
 
Analizar los determinantes de la felicidad a nivel país en 2026, evaluando el peso relativo del ingreso frente a otras variables del World Happiness Report. Integrar los datos del Global Mental Health Crisis Index 2026 para sumar el peso de variables relacionadas a la salud mental.
 
### Objetivos específicos
 
1. Cuantificar la correlación entre las distintas variables y el puntaje de felicidad.
2. Comparar el perfil completo de las variables entre los 10 países con mayor y menor puntaje de felicidad, para identificar en qué dimensiones se diferencian más.
3. Identificar países atípicos: aquellos con `gdp_per_capita` alto y `score` relativamente bajo, o `gdp_per_capita` bajo y `score` relativamente alto, y explorar qué otra variable explica esa desviación.
   
## Datasets
 
**World Happiness Update Report 2026** — Kaggle: [kaggle.com/datasets/jummanalnahian/world-happiness-update-report-2026](https://www.kaggle.com/datasets/jummanalnahian/world-happiness-update-report-2026)
 
Corte más reciente del World Happiness Report, con 9 columnas por país: `rank`, `country`, `gdp_per_capita`, `social_support`, `healthy_life_expectancy`, `freedom`, `generosity`, `corruption` y `score`.

**Global Mental Health Crisis Index 2026** — Kaggle: [kaggle.com/datasets/alitaqishah/global-mental-health-crisis-index-2026](https://www.kaggle.com/datasets/alitaqishah/global-mental-health-crisis-index-2026)

Archivo CSV con indicadores de salud mental por país tales como: `country`, `depression_pct`, `anxiety_pct`, `suicide_rate_per100k`, `treatment_gap_pct`, `youth_mh_crisis_score`, `mh_crisis_index`, `mh_system_score`.
 
