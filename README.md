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

## Descripción de variables

| Nombre de la columna | Tipo de dato | Descripción |
| :---- | :---- | :---- |
| `Country name` | Texto | Nombre del país al que corresponde la observación. |
| `year` | Entero | Año al que corresponde la información. Permite analizar la evolución de los indicadores a través del tiempo. |
| `Life Ladder` | Numérico | Evaluación que las personas realizan de su propia vida en una escala de 0 a 10\. 0 representa la peor vida posible y 10 la mejor vida posible. Es la principal medida de bienestar subjetivo utilizada por el World Happiness Report. |
| `Log GDP per capita` | Numérico | Logaritmo natural del PIB real per cápita, ajustado por paridad de poder adquisitivo. Representa el nivel económico de un país. |
| `Social support` | Numérico | Medida del apoyo social. Indica la proporción de personas que consideran que cuentan con familiares o amigos en quienes pueden confiar cuando tienen problemas. |
| `Healthy life expectancy at birth` | Numérico | Esperanza de vida saludable al nacer. Representa los años que se espera que una persona viva en condiciones de buena salud. |
| `Freedom to make life choices` | Numérico | Mide el grado en que las personas consideran que tienen libertad para elegir qué hacer con sus vidas. |
| `Generosity` | Numérico | Indicador relacionado con la generosidad de las personas, construido a partir de información sobre donaciones a organizaciones benéficas y ajustado por el nivel de PIB per cápita. |
| `Perceptions of corruption` | Numérico | Mide la percepción de corrupción en el gobierno y en las empresas. Valores más altos indican una mayor percepción de corrupción. |
| `Positive affect` | Numérico | Mide la frecuencia de emociones positivas experimentadas por las personas durante el día anterior, como disfrutar de la vida, reír o realizar actividades interesantes. |
| `Negative affect` | Numérico | Mide la frecuencia de emociones negativas experimentadas durante el día anterior, como preocupación, tristeza o enojo. |

