# Importando librerías
import pandas as pd
import numpy as np
import os 
from pathlib import Path
from typing import Dict, Union, Optional, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
#==================================# VAR #=====================================#

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileImportError(Exception):
    """Excepción personalizada para errores de importación de archivos."""
    pass

class DataReader(ABC):
    """Clase abstracta para lectura de datos siguiendo el principio Open/Closed."""
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        pass
    
    @abstractmethod
    def read(self, file_path: Path) -> pd.DataFrame:
        pass

class CSVReader(DataReader):
    """Implementación concreta para leer archivos CSV."""
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.csv'
    
    def read(self, file_path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise FileImportError(f"Error al leer el archivo CSV {file_path}: {str(e)}")

@dataclass
class ImportConfig:
    """Clase para configuración de importación."""
    allowed_extensions: tuple = ('.csv',)
    recursive: bool = True
    encoding: str = 'utf-8'

class DataImporter:
    """Clase principal para importación de datos siguiendo el principio Single Responsibility."""
    
    def __init__(self, reader: DataReader, config: Optional[ImportConfig] = None):
        """
        Inicializa el importador de datos.
        
        Args:
            reader: Implementación de DataReader para leer archivos
            config: Configuración opcional para la importación
        """
        self.reader = reader
        self.config = config or ImportConfig()
    
    def import_data(self, folder_path: Union[str, Path]) -> Dict:
        """
        Importa archivos desde una carpeta y sus subcarpetas, preservando la estructura de directorios
        en un diccionario anidado.
        
        Args:
            folder_path: Ruta a la carpeta principal desde la cual se deben importar los archivos.
        
        Returns:
            Dict: Diccionario anidado que refleja la estructura de carpetas con los datos importados.
        
        Raises:
            FileImportError: Si hay problemas al importar los archivos.
            ValueError: Si la ruta proporcionada no existe o no es un directorio.
        
        Example:
            >>> importer = DataImporter(CSVReader())
            >>> data = importer.import_data("ruta/a/carpeta")
            >>> # Estructura resultante:
            >>> # {
            >>> #     'carpeta1': {
            >>> #         'archivo1.csv': DataFrame,
            >>> #         'subcarpeta1': {
            >>> #             'archivo2.csv': DataFrame
            >>> #         }
            >>> #     }
            >>> # }
        """
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists() or not folder_path.is_dir():
                raise ValueError(f"La ruta {folder_path} no existe o no es un directorio")
            
            return self._process_directory(folder_path)
            
        except Exception as e:
            logger.error(f"Error durante la importación: {str(e)}")
            raise FileImportError(f"No se pudo completar la importación: {str(e)}")
    
    def _process_directory(self, directory: Path) -> Dict:
        """
        Procesa recursivamente un directorio y sus contenidos.
        
        Args:
            directory: Ruta del directorio a procesar.
        
        Returns:
            Dict: Estructura de datos con los archivos importados.
        """
        result = {}
        
        try:
            for item in directory.iterdir():
                if item.is_file() and self.reader.can_handle(item):
                    logger.info(f"Importando archivo: {item}")
                    result[item.name] = self.reader.read(item)
                elif item.is_dir() and self.config.recursive:
                    result[item.name] = self._process_directory(item)
            
            return result
        
        except Exception as e:
            logger.error(f"Error procesando directorio {directory}: {str(e)}")
            raise FileImportError(f"Error en el directorio {directory}: {str(e)}")

@dataclass
class PrintConfig:
    """Configuración para el formato de impresión."""
    indent_size: int = 2
    branch_vertical: str = "│"
    branch_horizontal: str = "──"
    branch_corner: str = "└"
    branch_tee: str = "├"
    show_dimensions: bool = True
    show_datatypes: bool = False

class DataPrinter(ABC):
    """Clase abstracta para imprimir diferentes tipos de datos."""
    @abstractmethod
    def can_handle(self, value: Any) -> bool:
        pass
    
    @abstractmethod
    def format_output(self, value: Any, config: PrintConfig) -> str:
        pass

class DataFramePrinter(DataPrinter):
    """Implementación para imprimir información de DataFrames."""
    def can_handle(self, value: Any) -> bool:
        return isinstance(value, pd.DataFrame)
    
    def format_output(self, value: pd.DataFrame, config: PrintConfig) -> str:
        info = f"(DataFrame: {value.shape[0]} filas, {value.shape[1]} columnas"
        if config.show_datatypes:
            info += f", dtypes: {', '.join(value.dtypes.astype(str))}"
        return info + ")"

class TreePrinter:
    """Clase principal para imprimir estructuras de datos en formato árbol."""
    
    def __init__(self, config: PrintConfig = None, data_printers: list[DataPrinter] = None):
        """
        Inicializa el impresor de árboles.
        
        Args:
            config: Configuración de formato de impresión
            data_printers: Lista de impresores para diferentes tipos de datos
        """
        self.config = config or PrintConfig()
        self.data_printers = data_printers or [DataFramePrinter()]
    
    def print_tree(self, data: Dict, indent: int = 0, is_last: bool = True, prefix: str = "") -> None:
        """
        Imprime la estructura de datos en formato árbol.
        
        Args:
            data: Diccionario o estructura de datos a imprimir
            indent: Nivel de indentación actual
            is_last: Indica si es el último elemento en el nivel actual
            prefix: Prefijo para la línea actual (para mantener las líneas verticales)
        
        Raises:
            ValueError: Si la entrada no es del tipo esperado
        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Los datos deben ser un diccionario")
            
            items = list(data.items())
            
            for i, (key, value) in enumerate(items):
                is_last_item = i == len(items) - 1
                self._print_node(key, value, indent, is_last_item, prefix)
                
        except Exception as e:
            logger.error(f"Error al imprimir el árbol: {str(e)}")
            print(f"Error: No se pudo imprimir la estructura completa - {str(e)}")
    
    def _print_node(self, key: str, value: Any, indent: int, is_last: bool, prefix: str) -> None:
        """
        Imprime un nodo individual del árbol.
        
        Args:
            key: Nombre del nodo
            value: Valor asociado al nodo
            indent: Nivel de indentación
            is_last: Indica si es el último elemento
            prefix: Prefijo para la línea actual
        """
        # Determina los caracteres a usar para las ramas
        branch = self.config.branch_corner if is_last else self.config.branch_tee
        
        # Construye la línea actual
        current_prefix = prefix + branch + self.config.branch_horizontal
        print(f"{prefix}{branch}{self.config.branch_horizontal} {key}")
        
        # Prepara el prefijo para los elementos hijos
        new_prefix = prefix
        if not is_last:
            new_prefix += self.config.branch_vertical + " " * self.config.indent_size
        else:
            new_prefix += " " * (self.config.indent_size + 1)
        
        # Procesa el valor según su tipo
        for printer in self.data_printers:
            if printer.can_handle(value):
                print(f"{new_prefix}{printer.format_output(value, self.config)}")
                return
        
        # Si es un diccionario, procesa recursivamente
        if isinstance(value, dict):
            self.print_tree(value, indent + self.config.indent_size, True, new_prefix)




#=====================# Funciones principales #==============================#
def import_data(ruta_carpeta):
    """
    Función wrapper para mantener compatibilidad con el código existente.
    
    Args:
        folder_path: Ruta a la carpeta principal.
    
    Returns:
        Dict: Estructura de datos con los archivos importados.
    """
    try:
        importer = DataImporter(CSVReader())
        result = importer.import_data(ruta_carpeta)
        logger.info("Importación completada exitosamente")
        return result
    except Exception as e:
        logger.error(f"Error en la importación: {str(e)}")
        raise


def imprimir_cascada(diccionario: Dict, indent: int = 0) -> None:
    """
    Función wrapper para mantener compatibilidad con el código existente.
    
    Args:
        diccionario: Diccionario anidado que contiene la estructura
        indent: Nivel de indentación inicial
    
    Example:
        >>> estructura = {
        ...     'carpeta1': {
        ...         'archivo1.csv': pd.DataFrame(...),
        ...         'subcarpeta1': {
        ...             'archivo2.csv': pd.DataFrame(...)
        ...         }
        ...     }
        ... }
        >>> imprimir_cascada(estructura)
        └── carpeta1
            ├── archivo1.csv
            │   (DataFrame: 100 filas, 5 columnas)
            └── subcarpeta1
                └── archivo2.csv
                    (DataFrame: 200 filas, 3 columnas)
    """
    try:
        printer = TreePrinter(PrintConfig(indent_size=indent))
        printer.print_tree(diccionario)
        logger.info("Estructura impresa exitosamente")
    except Exception as e:
        logger.error(f"Error al imprimir la estructura: {str(e)}")
        print(f"Error: No se pudo imprimir la estructura - {str(e)}")

def datetime_fix(dataframes):
    """
    Convierte la columna 'Local time' de todos los DataFrames en el diccionario 
    a un objeto datetime de pandas.

    Args:
        dataframes (dict): Un diccionario que contiene DataFrames organizados por pares de divisas y marcos de tiempo.
            La estructura esperada es:
            {
                'EURUSD': {
                    '1M': {
                        'EURUSD_1M_BID.csv': DataFrame,
                        'EURUSD_1M_ASK.csv': DataFrame
                    },
                    ...
                },
                ...
            }

    Returns:
        dict: El mismo diccionario de entrada con las columnas 'Local time' convertidas a tipo datetime.
    """
    # Iterar a través del diccionario
    for currency_pair, timeframes in dataframes.items():
        for timeframe, dfs in timeframes.items():
            for filename, df in dfs.items():
                # Verificar si la columna "Local time" existe en el DataFrame
                if 'Local time' in df.columns:
                    # Transformar la columna "Local time" a tipo datetime
                    try:
                        df['Local time'] = pd.to_datetime(df['Local time'], format='%d.%m.%Y %H:%M:%S.%f GMT%z')
                        # (Opcional) Imprimir un mensaje para verificar la conversión
                        #print(f"Converted 'Local time' in {filename} under {currency_pair} {timeframe}")
                    except Exception as e:
                        print(f"Error converting 'Local time' in {filename} under {currency_pair} {timeframe}: {e}")
    print('AJuste de fecha completa')
    return dataframes

def dropna_all(dataframes):
    # Iterar a través del diccionario
    for currency_pair, timeframes in dataframes.items():
        for timeframe, dfs in timeframes.items():
            for filename, df in dfs.items():
                # Eliminar filas con valores faltantes
                df_cleaned = df.dropna()

                # (Opcional) Imprimir un mensaje para verificar el cambio
                #print(f"Dropped NaN values in {filename} under {currency_pair} {timeframe}. Rows before: {len(df)}, Rows after: {len(df_cleaned)}")

                # Reemplazar el DataFrame original con el DataFrame limpio
                dfs[filename] = df_cleaned
    print('Clear completed')
    return dataframes

def validate_dataframes(dataframes):
    for currency_pair, timeframes in dataframes.items():
        for timeframe, dfs in timeframes.items():
            for filename, df in dfs.items():
                # Verificar el número de columnas
                if df.shape[1] != 6:
                    print(f"Error: {filename} under {currency_pair} {timeframe} has {df.shape[1]} columns (expected 6).")
                    continue  # Salta a la siguiente iteración

                # Verificar los tipos de datos
                if not pd.api.types.is_datetime64_any_dtype(df['Local time']):
                    print(f"Error: 'Local time' in {filename} is not of type datetime.")
                
                # Verificar que las demás columnas sean float o int
                for col in df.columns:
                    if col != 'Local time':
                        if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])):
                            print(f"Error: Column '{col}' in {filename} is not of type float or int.")
    
    print("Validation complete.")
# Función principal
def main():
    # Especifica la ruta a la carpeta que contiene los archivos CSV
    ruta_carpeta = "/home/KAISER/Documents/HermesD_B/HermesD_B/data/raw"
    
    # Importamos las bases de datos
    DB = import_data(ruta_carpeta)
    
    # Imprimir la arquitectura del diccionario como cascada
    imprimir_cascada(DB)

    #Arreglando la columna Local time de todos los arhivos
    DB = datetime_fix(DB)
    print(DB['EURUSD']['1H']['EURUSD_1H_BID.csv'])

    #=====================# lIMPIEZA #=====================#
    #QUitando los datos dropna y faltante
    DB = dropna_all(DB)

    #=====================# Rectificia y comprobador  #=====================#
    #vALIDANDO FORMA Y ESENCIA DEL DATAFRAME
    validate_dataframes(DB)
main()
