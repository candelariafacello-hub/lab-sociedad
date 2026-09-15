import os 
os.getcwd()

from pathlib import Path
import pandas as pd

# Ruta raíz del proyecto
BASE_DIR = Path.cwd()

# Ruta relativa 
archivo = "world_happiness_Update_report_2026.csv"
ruta_csv = BASE_DIR / "lab-sociedad" / "data" / archivo

# Verificar que el archivo realmente existe antes de cargarlo
if not ruta_csv.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {ruta_csv}")

# Cargar el archivo CSV
df = pd.read_csv(ruta_csv)

# Inspección inicial de los datos
print(f"--- Archivo cargado exitosamente desde: {ruta_csv.name} ---")
print(df.info())
print("\nPrimeras 5 filas:")
print(df.head())

## Dimensiones y estructura de los datos
df.info()
df.shape

# Datos faltantes
df.isnull().sum()  # Conteo de NA por columna
(df == "").sum()   # Falsos nulos (cadenas vacías)

## 5.4 Duplicados
df.duplicated().sum() # Cantidad de filas exactamente iguales
# %%
