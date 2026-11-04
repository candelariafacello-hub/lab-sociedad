Partimos de ¿qué factores explican realmente la felicidad de un país? Nos pusimos tres objetivos concretos:

Cuantificar la correlación de cada variable con el puntaje.
Comparar el perfil de los 10 países más y menos felices.
Identificar países atípicos, por ejemplo con PIB alto y felicidad baja, o al revés, y ver qué variable explica la desviación.

La idea era medir cuan influyente es el ingreso frente al resto de las variables del  World Happiness Update Report 2026, y después sumar salud mental.

**Limpieza de World Hapiness Update Report 2026.**
- En el dataset de World Hapiness Update Report 2026, no encontramos valores nulos pero si filas con países duplicados. Decidimos promediar los países repetidos y no quedarnos con una fila ni descartarlas. Con eso conservamos toda la información y llegamos a una fila por país, que es lo que necesitábamos para unir después con otro dataset y para que cada país sea único en las correlaciones.
- Renombramos las columnas justo después de limpiar, con un diccionario central. Hacerlo temprano evitó mezclar nombres en inglés y español en el resto del trabajo. También definimos atajos para las columnas que más se repiten (país y puntaje)

**Unión con el dataset con Global Mental Health Crisis Index 2026.** 
Solo las variables de salud mental relevantes. Del dataset de salud mental conservamos depresión, ansiedad, suicidios, brecha de tratamiento, crisis juvenil, índice de crisis y calidad del sistema. El resto no aportaba a la pregunta.
Unión que descarta países. Usamos un inner join, así que los países sin datos de salud mental quedan afuera. Lo dejamos explícito en el script como una pérdida asumida, porque el dataset de salud mental tiene menos países.

**Gráficos de dispersión con recta y países atípicos**
- Cada variable se cruza con el puntaje mediante una recta de regresión y el coeficiente de correlación de Pearson. También detectamos los países que más se alejan de la recta, los países que “no cumplen” la regla general.
- Armamos también graficos para las variables del World Happiness Update Report y otro para salud mental, ordenados de menor a mayor. Dan una lectura rápida de qué factores se asocian más con la felicidad.

**Promedios de variables en paises TOP Y BOTTOM 10**
Comparamos los promedios de todas las variables entre los diez países más felices y los diez menos felices. Es una forma simple de ver cómo se diferencian los extremos.

**Análisis por región**
A cada país de la tabla, averiguamos en qué continente está y lo guardamos en una columna nueva (“Región”). Esa columna es la que permite después agrupar los países y armar el mapa y los gráficos por región. Como no reconoce algunos nombres (Kosovo, Taiwán, Hong Kong, Somalilandia, los dos Congos, Türkiye, entre otros), los corregimos a mano con un diccionario. Los países que seguían sin región se eliminan.

**Mapa por región.** 
Cada país toma el valor promedio de su región y se pinta según la felicidad promedio. Es un mapa de regiones, no de países individuales, y  muestra también depresión, ansiedad e índice de crisis. Se guarda en HTML para no depender de otras librerías.

**Modelo predictivo del puntaje**
Usamos los mismos países para todos los modelos. Nos quedamos solo con países con todos los datos completos, para que la comparación entre modelos sea justa.

El modelo de referencia.
Muestra el promedio de todos. Es la forma más simple de predecir.
Si un modelo más sofisticado no le gana a esa respuesta, entonces no sirve. En el gráfico de resultados, ese modelo dio negativo, por lo que adivinar el promedio es peor que no tener ninguna idea, y eso confirma que los otros modelos sí aportan.

Una regresión lineal, simple e interpretable. Cada uno se prueba con solo las variables del WHR y con WHR más salud mental. La pregunta central es si la salud mental mejora la predicción de la felicidad respecto de los factores tradicionales.

Para interpretar, usamos los coeficientes de la regresión lineal con variables estandarizadas. Cada coeficiente indica cuánto cambia el puntaje por cada desviación estándar de la variable, y eso permite compararlas entre sí.

**Países que sorprenden. **
Hicimos predicciones donde cada país se predice con un modelo que no lo vio. Luego listamos los cinco países más felices y los cinco menos felices de lo que el modelo esperaba. Muestran lo que las variables no capturan, como cultura o contexto.

Hicimos una simulación “¿y si mejorara X?”. Armamos una función que estima cuánto cambiaría el puntaje de un país si una variable subiera una desviación estándar. 


