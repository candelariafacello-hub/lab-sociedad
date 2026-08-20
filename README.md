# Felicidad, Ingreso y Bienestar: ¿Qué factores explican realmente la felicidad de un país?

Introducción a Ciencia de Datos – Candelaria Facello, Belen Etchegoimberry, Camila Barboza y Clara Gimenez. 

## Problema a resolver 

Existe una creencia extendida de que el ingreso económico es el principal determinante de la felicidad de una población. Sin embargo, el *World Happiness Report* mide la felicidad como una combinación de varios factores: ingreso per cápita, apoyo social, expectativa de vida saludable, libertad para tomar decisiones de vida, generosidad y percepción de corrupción.

**Problema a resolver:** determinar en qué medida el ingreso per cápita explica la felicidad de un país en comparación con otros factores (apoyo social, libertad, salud, corrupción), y si existen patrones regionales o temporales que se aparten de esta relación. 

## Objetivos

Analizar los determinantes de la felicidad a nivel país entre 2013 y 2025, evaluando el peso relativo del ingreso frente a otras variables del World Happiness Report.

### Objetivos específicos

1. Describir la evolución del ranking de felicidad global y de países/regiones de interés (ej. América Latina) entre 2013 y 2025\.  
2. Cuantificar la correlación entre ingreso (GDP per cápita) y puntaje de felicidad, y compararla con la correlación de otras variables (apoyo social, libertad, salud, corrupción, generosidad).  
3. Identificar países "atípicos": aquellos con ingreso alto y felicidad relativamente baja, o ingreso bajo y felicidad relativamente alta, y explorar qué otras variables explican esas desviaciones.

## Datasets

Vamos a utilizar el dataset **World Happiness Index by Reports \[2013-2023\]**, disponible en Kaggle: [kaggle.com/datasets/simonaasm/world-happiness-index-by-reports-2013-2023](https://www.kaggle.com/datasets/simonaasm/world-happiness-index-by-reports-2013-2023).

Contiene el puntaje de felicidad de distintos países a lo largo de 10 años (2013-2023), junto con los seis subíndices que componen ese puntaje según el World Happiness Report: GDP per cápita, apoyo social, expectativa de vida saludable, libertad para tomar decisiones de vida, generosidad y percepción de corrupción. 

