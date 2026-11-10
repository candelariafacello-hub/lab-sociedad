Decisiones · MD

Partimos de ¿qué factores explican realmente la felicidad de un país? 
 
La idea era medir cuan influyente son las variables del World Happiness Report 2026 y después se nos ocurrió sumar salud mental.

 
**Limpieza de World Hapiness Update Report 2026.**

- En el dataset de World Hapiness Update Report 2026, no encontramos valores nulos pero si filas con países duplicados. Decidimos promediar los países repetidos y no quedarnos con una fila ni descartarlas. Con eso conservamos toda la información y llegamos a una fila por país, que es lo que necesitábamos para unir después con otro dataset y para que cada país sea único en las correlaciones.
- Antes de promediar sacamos la columna del ranking, porque un ranking no tiene sentido promediarlo.
- Renombramos las columnas justo después de limpiar, con un diccionario. Hacerlo temprano evitó mezclar nombres en inglés y español en el resto del trabajo. También definimos atajos para las columnas que más se repiten (país y puntaje).

  
**Gráficos de correlación entre las variables del WHR y felicidad**

- Cada variable se cruza con el puntaje mediante una recta de regresión y el coeficiente de correlación de Pearson. También detectamos los países que más se alejan de la recta, los países que “no cumplen” la regla general.
- Los atípicos son los 5 países con mayor residuo (distancia vertical a la recta, en valor absoluto), y se calculan por separado para cada variable. Las etiquetas se ubican según dónde cae el punto (izquierda o derecha, arriba o abajo de la recta) para que no se pisen con el borde.
- Armamos una sola función para todos los gráficos, así las variables del WHR (en azul) y las de salud mental (en verde) se grafican igual y se pueden comparar.
- Además de los gráficos, imprimimos una tabla con la correlación de cada variable con el puntaje.


**Por qué sumamos salud mental**

Las variables del WHR dieron correlaciones muy altas con el puntaje (cerca de 0,9 o más, salvo generosidad, aprox 0,25). Eso no es un hallazgo: son los criterios con los que el WHR construye y explica el puntaje. Por eso buscamos una fuente externa que pudiera aportar información nueva, y elegimos el Global Mental Health Crisis Index 2026.

 
**Unión con el dataset con Global Mental Health Crisis Index 2026.** 

Antes de unir revisamos nulos y países repetidos en este dataset, y no había ninguno de los dos.
Solo las variables de salud mental relevantes. Del dataset de salud mental conservamos depresión, ansiedad, suicidios, brecha de tratamiento, crisis juvenil, índice de crisis y calidad del sistema. El resto no aportaba a la pregunta. También las renombramos al español con otro diccionario.
Usamos un inner join, así que los países sin datos de salud mental quedan afuera. Después de unir verificamos que no quedaran países duplicados.

 
**Gráficos de barras de resumen de las correlaciones**

Armamos un gráfico para las variables del WHR y otro para salud mental, ordenados de menor a mayor. Dan una lectura rápida de qué factores se asocian más con la felicidad. La función también imprime cuál es la variable más y menos asociada al puntaje de cada grupo (en valor absoluto).
 
Las variables de salud mental se separaron en tres grupos:
- Correlación negativa fuerte: índice de crisis, crisis juvenil y brecha de tratamiento (aprox -0,8).
- Correlación positiva fuerte: calidad del sistema de salud mental (aprox 0,8).
- Correlación débil: ansiedad (casi cero), suicidios (aprox -0,2) y depresión, que da positiva (aprox 0,45).
Lo que parece importar no es cuánta depresión o ansiedad hay, sino cómo responde el país (calidad del sistema y acceso a tratamiento). Por eso decidimos profundizar en las variables de salud mental.

 
**Promedios de variables en paises TOP y BOTTOM 10**

Comparamos los promedios de todas las variables entre los diez países más felices y los diez menos felices. Es una forma simple de ver cómo se diferencian los extremos.
- Hicimos dos gráficos (WHR y salud mental) porque las variables tienen rangos muy distintos.
- Las 6 variables del WHR son más altas en los países más felices, con las mayores diferencias en PIB per cápita y percepción de corrupción. Generosidad es la que menos separa.
- En salud mental, depresión y ansiedad son casi iguales en los dos grupos. Lo que sí cambia es la respuesta del sistema: la brecha de tratamiento pasa de alrededor de 29% a 90% y la calidad del sistema de 69 a 7. Una posible explicación es que donde no hay sistema de salud mental la depresión se diagnostica y registra menos, y por eso el porcentaje parece bajo.

  
**Agrupamiento por región**

Para cada país de la tabla, averiguamos en qué continente está y lo guardamos en una columna nueva (“Región”). Esa columna es la que permite después agrupar los países y armar el mapa y los gráficos por región. Usamos la librería pycountry_convert, que necesita los nombres de los países en inglés (por eso renombramos las columnas pero no los valores). Como no reconoce algunos nombres (Kosovo, Taiwán, Hong Kong, Somalilandia, los dos Congos, Türkiye, entre otros), los corregimos a mano con un diccionario. Los países que seguían sin región se eliminan, y los imprimimos para saber cuáles se pierden.

 
**Mapa interactivo por región** 

Cada país toma el valor promedio de su región y se pinta según la felicidad promedio. Es un mapa de regiones, no de países individuales, y  muestra también depresión, ansiedad e índice de crisis. Se guarda en HTML para no depender de otras librerías, y además se abre en el navegador al correr el script.

 
**Desempeño por región**

Para comparar cada región con el promedio de los países usamos el puntaje z (cuántos desvíos estándar se aleja del promedio), así todas las variables quedan en la misma escala.
- En las variables donde un valor más alto es peor (depresión, ansiedad, suicidios, brecha, crisis juvenil e índice de crisis) invertimos el signo, para que positivo siempre signifique “mejor que el promedio”.
- Hicimos un gráfico por región, con verde para lo que está por encima del promedio y rojo para lo que está por debajo, e indicamos cuántos países tiene cada una (n).
- Europa está por encima del promedio en casi todo, y sus mayores fortalezas son de salud mental, incluso por encima del PIB. África es el peor caso, por debajo en casi todas las variables. Asia queda cerca del promedio, con su punto más débil en el sistema de salud mental y la brecha de tratamiento. América del Sur queda algo por debajo, con la ansiedad como mayor debilidad.
- América del Norte (n=4) y Oceanía (n=2) tienen muy buen desempeño, pero son pocos países, así que sus promedios no son tan confiables.

Como el desempeño cambia tanto de una región a otra, decidimos sumar la región a los modelos.


**Modelo predictivo del puntaje**

Usamos los mismos países para todos los modelos. Nos quedamos solo con países con todos los datos completos, para que la comparación entre modelos sea justa.
 
Decidimos hacer una regresión lineal que estime el puntaje con una fórmula del tipo Felicidad = a + b1·x1 + b2·x2 + ... 

Para comparar modelos usamos el R², que indica qué parte de las diferencias de felicidad entre países explica el modelo (0 = no mejora nada, 1 = predice perfecto).

Como el R² sube siempre que se agrega una variable, aunque no aporte nada, también miramos el R² ajustado, que penaliza las variables que no suman. Las decisiones las tomamos con el R² ajustado.

Un R² alto muestra que las variables están asociadas a la felicidad, no que la causen.
 
Armamos dos funciones: 
una que ajusta el modelo con las variables que le pasemos, y otra que recibe varios modelos y devuelve una tabla con su R² y R² ajustado. Así comparar modelos es solo agregar una línea.
 
Primero, el control con el WHR.
Probamos solo apoyo social y las 6 variables del WHR. Como esperábamos, el R² da casi 1. Esto confirma lo que vimos en las correlaciones: no es un descubrimiento, porque el puntaje se construye con esas variables. Por eso las dejamos de lado y nos enfocamos en las de salud mental.
 
Elección de las variables de salud mental.
Usamos lo que vimos en la exploración:
- Brecha de tratamiento, calidad del sistema y crisis juvenil: correlaciones fuertes y fáciles de interpretar.
- Depresión: da positiva, al revés de lo esperado. La probamos para ver si aporta algo cuando ya están las otras.
- Quedan afuera ansiedad (sin relación lineal), suicidios (relación débil y arrastrada por 2 países extremos) y el índice de crisis. El índice es compuesto, armado a partir de otros indicadores de salud mental, así que combinarlo con ellos sería repetir información. Lo usamos solo, como referencia, y no lo elegimos para el modelo final porque no sabemos cómo está calculado y no podríamos interpretar su fórmula.
  
Antes de combinarlas miramos la correlación entre ellas, y marcamos los pares con |r| > 0,8: si dos variables dan casi la misma información, usarlas juntas casi no mejora el modelo.

Tabla de R² con salud mental:
Probamos cada variable sola y después combinaciones, para ver cuáles suman. Elegimos automáticamente la combinación con mayor R² ajustado. Sin usar ninguna variable del WHR, la salud mental ya explica una parte importante de las diferencias de felicidad.
 
Región y PIB como control:
- La región entra como variables 0/1 (vale 1 si el país es de esa región y 0 si no), con Europa como base. El coeficiente de cada región es la diferencia de felicidad respecto a Europa. Oceanía y América del Norte tienen pocos países, así que sus coeficientes son poco confiables.
- El PIB no lo usamos para explicar la felicidad (es una variable del WHR), sino como control. Los países ricos suelen tener mejores sistemas de salud mental, así que queríamos saber si la salud mental aporta algo más allá de la riqueza. Comparamos el R² ajustado de “solo PIB” con “salud mental + PIB”: si sube más de 0,02, consideramos que la salud mental aporta información que el PIB no captura.


**Modelo final: salud mental + región**

Es el modelo hacia el que nos guió el análisis. No incluimos el PIB porque es una variable del WHR y queríamos que el modelo se enfoque en la salud mental. Mostramos los coeficientes con su p-valor (p < 0,05 indica evidencia de que la variable está asociada a la felicidad) y la fórmula completa de la regresión, con nombres cortos para que se pueda leer.  Si el modelo fuera perfecto, todos los puntos quedarían sobre la línea roja. Los puntos se pintan por región y la fórmula va debajo del gráfico. Etiquetamos los cinco países más alejados de lo que predice el modelo, y listamos los tres más felices y los tres menos felices de lo que el modelo esperaba. Muestran lo que las variables no capturan, como cultura o contexto.
