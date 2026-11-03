Partimos de ¿qué factores explican realmente la felicidad de un país? Nos pusimos tres objetivos concretos:

Cuantificar la correlación de cada variable con el puntaje.
Comparar el perfil de los 10 países más y menos felices.
Identificar países atípicos, por ejemplo con PIB alto y felicidad baja, o al revés, y ver qué variable explica la desviación.

La idea era medir cuan influyente es el ingreso frente al resto de las variables del  World Happiness Update Report 2026, y después sumar salud mental.

**Limpieza de World Hapiness Update Report 2026.**
- En el dataset de World Hapiness Update Report 2026, no encontramos valores nulos pero si filas con países duplicados. Decidimos promediar los países repetidos y no quedarnos con una fila ni descartarlas. Con eso conservamos toda la información y llegamos a una fila por país, que es lo que necesitábamos para unir después con otro dataset y para que cada país sea único en las correlaciones.
- Renombramos las columnas justo después de limpiar, con un diccionario central. Hacerlo temprano evitó mezclar nombres en inglés y español en el resto del trabajo. También definimos atajos para las columnas que más se repiten (país y puntaje)

**Unión con el dataset con Global Mental Health Crisis Index 2026.** 
Conservamos solo las variables de salud mental relevantes: 
        "country",
        "depression_pct",
        "anxiety_pct",
        "suicide_rate_per100k",
        "treatment_gap_pct",
        "youth_mh_crisis_score",
        "mh_crisis_index",
        "mh_system_score"

