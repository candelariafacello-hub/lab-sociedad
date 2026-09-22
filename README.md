

Readme · MD
# Felicidad y bienestar: ¿qué factores explican realmente la felicidad de un país?
 
Introducción a Ciencia de Datos – Candelaria Facello, Belen Etchegoimberry, Camila Barboza y Clara Gimenez.
 
## Descripción del problema
 
Existe una creencia extendida de que el ingreso económico es el principal determinante de la felicidad de una población. Sin embargo, el *World Happiness Report* mide la felicidad como una combinación de varios factores: ingreso per cápita, apoyo social, expectativa de vida saludable, libertad para tomar decisiones de vida, generosidad y percepción de corrupción.
 
**Problema a resolver:** determinar, para la edición más reciente del World Happiness Report, en qué medida el ingreso per cápita explica el puntaje de felicidad de un país en comparación con las otras cinco variables que componen el índice (apoyo social, salud, libertad, generosidad y corrupción), e identificar los casos de países cuya felicidad se aparta de lo esperado según su nivel de ingreso.
 
## Objetivos
 
### Objetivo general
 
Analizar los determinantes de la felicidad a nivel país en 2026, evaluando el peso relativo del ingreso frente a otras variables del World Happiness Report.
 
### Objetivos específicos
 
1. Cuantificar la correlación entre `gdp_per_capita` y `score`, y compararla con la correlación de las demás variables (`social_support`, `healthy_life_expectancy`, `freedom`, `generosity`, `corruption`).
2. Identificar países atípicos: aquellos con `gdp_per_capita` alto y `score` relativamente bajo, o `gdp_per_capita` bajo y `score` relativamente alto, y explorar qué otra variable explica esa desviación.
3. Comparar el perfil completo de las 6 variables entre los 10 países con mayor y menor puntaje de felicidad, para identificar en qué dimensiones se diferencian más.
4. Agrupar países según su perfil en las 6 variables y evaluar características en común dentro de su grupo.
## Dataset
 
**World Happiness Update Report 2026** — Kaggle: [kaggle.com/datasets/jummanalnahian/world-happiness-update-report-2026](https://www.kaggle.com/datasets/jummanalnahian/world-happiness-update-report-2026)
 
Corte más reciente del World Happiness Report, con 9 columnas por país: `rank`, `country`, `gdp_per_capita`, `social_support`, `healthy_life_expectancy`, `freedom`, `generosity`, `corruption` y `score`.
 