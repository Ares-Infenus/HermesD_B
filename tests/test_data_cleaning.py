# Importando librerías
import pandas as pd
import numpy as np
import os 

# Función para importar datos
def import_data(ruta_carpeta):
    estructura = {}
    
    # Recorre todas las subcarpetas y archivos
    for root, dirs, files in os.walk(ruta_carpeta):
        # Calcula la ruta relativa de la carpeta
        ruta_relativa = os.path.relpath(root, ruta_carpeta)
        
        # Si la ruta relativa es solo '.', significa que estamos en la carpeta raíz
        if ruta_relativa == '.':
            current_dict = estructura
        else:
            # Navega hasta el nivel actual en el diccionario
            current_dict = estructura
            for part in ruta_relativa.split(os.sep):
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
        
        # Agrega los archivos CSV a la estructura
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                current_dict[file] = pd.read_csv(file_path)

    return estructura

# Función para imprimir la arquitectura del diccionario como una cascada
def imprimir_cascada(diccionario, indent=0):
    for key, value in diccionario.items():
        print('    ' * indent + f"└── {key}")
        if isinstance(value, pd.DataFrame):
            print('    ' * (indent + 1) + f"(DataFrame: {value.shape[0]} filas, {value.shape[1]} columnas)")
        else:
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
