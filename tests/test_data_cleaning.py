# Importando librerías
import pandas as pd
import numpy as np
import os 

# Función para importar datos
def import_data(ruta_carpeta):
    """
    Importa archivos CSV desde una carpeta y sus subcarpetas, preservando la estructura de directorios
    en un diccionario anidado. Cada carpeta se convierte en una clave del diccionario y los archivos CSV
    se cargan como DataFrames de pandas.

    Parámetros:
    ruta_carpeta (str): La ruta a la carpeta principal desde la cual se deben importar los archivos CSV.

    Retorna:
    dict: Un diccionario anidado que refleja la estructura de carpetas. Cada subcarpeta se representa como
          un subdiccionario, y los archivos CSV se almacenan como DataFrames.

    Ejemplo de estructura de salida:
    {
        'carpeta1': {
            'archivo1.csv': DataFrame,
            'subcarpeta1': {
                'archivo2.csv': DataFrame
            }
        },
        'carpeta2': {
            'archivo3.csv': DataFrame
        }
    }
    """

    # Inicializa el diccionario que almacenará la estructura de carpetas y los archivos CSV
    estructura = {}
    
    # Recorre todas las carpetas y archivos en la ruta especificada
    for root, dirs, files in os.walk(ruta_carpeta):
        # Obtiene la ruta relativa para identificar en qué nivel de la jerarquía se encuentra cada carpeta
        ruta_relativa = os.path.relpath(root, ruta_carpeta)
        
        # Determina el diccionario actual donde se deben almacenar los archivos y carpetas
        if ruta_relativa == '.':
            # Si estamos en la carpeta raíz, usa el diccionario principal
            current_dict = estructura
        else:
            # Si estamos en una subcarpeta, navega hasta el nivel correspondiente en el diccionario
            current_dict = estructura
            for part in ruta_relativa.split(os.sep):  # Divide la ruta en cada carpeta
                if part not in current_dict:
                    # Si la subcarpeta no existe en el diccionario, crea una nueva clave
                    current_dict[part] = {}
                # Avanza al subdiccionario correspondiente
                current_dict = current_dict[part]
        
        # Agrega archivos CSV encontrados en la carpeta actual al diccionario
        for file in files:
            if file.endswith(".csv"):  # Verifica que el archivo tenga extensión CSV
                # Obtiene la ruta completa del archivo
                file_path = os.path.join(root, file)
                # Lee el archivo CSV en un DataFrame de pandas y lo almacena en el diccionario actual
                current_dict[file] = pd.read_csv(file_path)

    return estructura


def imprimir_cascada(diccionario, indent=0):
    """
    Imprime la estructura de un diccionario anidado en formato de cascada, simulando una jerarquía de carpetas.
    Si un valor en el diccionario es un DataFrame de pandas, muestra la cantidad de filas y columnas.

    Parámetros:
    diccionario (dict): El diccionario anidado que contiene la estructura de carpetas y archivos.
    indent (int): Nivel de indentación actual, usado para controlar el sangrado en la impresión.
                  Por defecto es 0 y aumenta recursivamente en cada nivel de anidación.

    Ejemplo de estructura de salida:
    └── carpeta1
        ├── archivo1.csv
        │   (DataFrame: 100 filas, 5 columnas)
        └── subcarpeta1
            └── archivo2.csv
                (DataFrame: 200 filas, 3 columnas)
    """

    # Itera a través de las claves y valores en el diccionario
    for key, value in diccionario.items():
        # Imprime la clave con un formato en cascada, usando indentación basada en el nivel actual
        print('    ' * indent + f"└── {key}")

        # Verifica si el valor es un DataFrame
        if isinstance(value, pd.DataFrame):
            # Si es un DataFrame, imprime el número de filas y columnas
            print('    ' * (indent + 1) + f"(DataFrame: {value.shape[0]} filas, {value.shape[1]} columnas)")
        else:
            # Si el valor es un subdiccionario, llama a la función de manera recursiva aumentando la indentación
            imprimir_cascada(value, indent + 1)


# Función principal
def main():
    # Especifica la ruta a la carpeta que contiene los archivos CSV
    ruta_carpeta = "/home/KAISER/Documents/HermesD_B/HermesD_B/data/raw"
    
    # Importamos las bases de datos
    DB = import_data(ruta_carpeta)
    
    # Imprimir la arquitectura del diccionario como cascada
    imprimir_cascada(DB)

    

main()
