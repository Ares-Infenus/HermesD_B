# Importando librerías
import pandas as pd
import numpy as np
import os 
from pathlib import Path
from typing import Dict, Union, Optional, Any, List
import logging
from dataclasses import dataclass
import pytz
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


class DateTimeConfig:
    """Configuración para el procesamiento de fechas."""
    default_format: str = '%d.%m.%Y %H:%M:%S.%f GMT%z'
    timezone: str = 'UTC'
    column_name: str = 'Local time'
    drop_bad_rows: bool = False
    coerce_errors: bool = True

class DataFrameProcessor(ABC):
    """Clase abstracta base para procesadores de DataFrames."""
    @abstractmethod
    def can_process(self, df: pd.DataFrame) -> bool:
        pass
    
    @abstractmethod
    def process(self, df: pd.DataFrame, config: Any) -> pd.DataFrame:
        pass

class DateTimeProcessor(DataFrameProcessor):
    """Procesador específico para conversión de fechas en DataFrames."""
    
    def can_process(self, df: pd.DataFrame) -> bool:
        """Verifica si el DataFrame puede ser procesado."""
        return isinstance(df, pd.DataFrame) and self.config.column_name in df.columns
    
    def __init__(self, config: Optional[DateTimeConfig] = None):
        """
        Inicializa el procesador de fechas.
        
        Args:
            config: Configuración para el procesamiento de fechas
        """
        self.config = config or DateTimeConfig()
    
    def process(self, df: pd.DataFrame, path_info: Dict[str, str]) -> pd.DataFrame:
        """
        Procesa las fechas en el DataFrame.
        
        Args:
            df: DataFrame a procesar
            path_info: Información sobre la ubicación del DataFrame
        
        Returns:
            pd.DataFrame: DataFrame con las fechas procesadas
        
        Raises:
            ValueError: Si hay errores en el procesamiento de fechas
        """
        try:
            df = df.copy()
            col_name = self.config.column_name
            
            if not self.can_process(df):
                logger.warning(f"La columna {col_name} no existe en el DataFrame")
                return df
            
            # Procesar fechas con manejo de errores
            try:
                df[col_name] = pd.to_datetime(
                    df[col_name],
                    format=self.config.default_format,
                    errors='coerce' if self.config.coerce_errors else 'raise'
                )
                
                # Manejar zonas horarias
                if self.config.timezone:
                    df[col_name] = df[col_name].dt.tz_convert(self.config.timezone)
                
                # Opcionalmente eliminar filas con fechas inválidas
                if self.config.drop_bad_rows:
                    invalid_dates = df[col_name].isna()
                    if invalid_dates.any():
                        logger.warning(
                            f"Se encontraron {invalid_dates.sum()} fechas inválidas en "
                            f"{path_info.get('filename', 'unknown file')}"
                        )
                        df = df.dropna(subset=[col_name])
                
                logger.info(
                    f"Fechas procesadas exitosamente en {path_info.get('filename', 'unknown file')}"
                )
                
            except Exception as e:
                logger.error(
                    f"Error al procesar fechas en {path_info.get('filename', 'unknown file')}: {str(e)}"
                )
                raise ValueError(f"Error en el procesamiento de fechas: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error general en el procesamiento: {str(e)}")
            raise

class DataFrameTraverser:
    """Clase para recorrer estructuras anidadas de DataFrames."""
    
    def __init__(self, processor: DataFrameProcessor):
        """
        Inicializa el traverser.
        
        Args:
            processor: Procesador a utilizar en cada DataFrame
        """
        self.processor = processor
    
    def traverse_and_process(self, data: Dict, current_path: Dict[str, str] = None) -> Dict:
        """
        Recorre la estructura de datos y procesa cada DataFrame encontrado.
        
        Args:
            data: Estructura de datos anidada
            current_path: Información sobre la ruta actual en la estructura
        
        Returns:
            Dict: Estructura procesada
        """
        if current_path is None:
            current_path = {}
        
        try:
            result = {}
            
            for key, value in data.items():
                new_path = {**current_path, 'current_key': key}
                
                if isinstance(value, pd.DataFrame):
                    new_path['filename'] = key
                    result[key] = self.processor.process(value, new_path)
                elif isinstance(value, dict):
                    result[key] = self.traverse_and_process(value, new_path)
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Error al recorrer la estructura: {str(e)}")
            raise

@dataclass
class CleaningConfig:
    """Configuración para la limpieza de datos."""
    drop_all_na: bool = True
    drop_any_na: bool = False
    drop_duplicates: bool = False
    remove_outliers: bool = False
    threshold: Optional[float] = None
    subset: Optional[List[str]] = None
    log_details: bool = True

class DataCleaner:
    """Procesador específico para limpieza de DataFrames."""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Inicializa el limpiador de datos.
        
        Args:
            config: Configuración para la limpieza de datos
        """
        self.config = config if config is not None else CleaningConfig()
    
    def clean_dataframe(self, df: pd.DataFrame, path_info: Dict[str, str]) -> pd.DataFrame:
        """
        Limpia el DataFrame eliminando valores nulos según la configuración.
        
        Args:
            df: DataFrame a limpiar
            path_info: Información sobre la ubicación del DataFrame
        
        Returns:
            pd.DataFrame: DataFrame limpio
        """
        try:
            original_rows = len(df)
            df_cleaned = df.copy()
            
            # Aplicar limpieza
            df_cleaned = df_cleaned.dropna()
            
            # Registrar resultados si está habilitado el logging
            if self.config.log_details:
                rows_removed = original_rows - len(df_cleaned)
                if rows_removed > 0:
                    logger.info(
                        f"Limpieza completada en {path_info.get('filename', 'unknown file')}. "
                        f"Filas eliminadas: {rows_removed} ({(rows_removed/original_rows)*100:.2f}%)"
                    )
            
            return df_cleaned
            
        except Exception as e:
            logger.error(f"Error en la limpieza del DataFrame: {str(e)}")
            return df  # Retorna el DataFrame original en caso de error

class ValidationError(Exception):
    """Excepción personalizada para errores de validación."""
    pass

@dataclass
class ValidationConfig:
    """Configuración para la validación de DataFrames."""
    expected_columns: int = 6
    datetime_column: str = 'Local time'
    expected_numeric_types: tuple = (np.float64, np.int64)
    strict_validation: bool = True

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

def datetime_fix(dataframes: Dict) -> Dict:
    """
    Función wrapper para mantener compatibilidad con el código existente.
    
    Args:
        dataframes: Diccionario con la estructura de datos a procesar
    
    Returns:
        Dict: Estructura de datos procesada
    
    Example:
        >>> data = {
        ...     'EURUSD': {
        ...         '1M': {
        ...             'EURUSD_1M_BID.csv': df1,
        ...             'EURUSD_1M_ASK.csv': df2
        ...         }
        ...     }
        ... }
        >>> processed_data = datetime_fix(data)
    """
    try:
        processor = DateTimeProcessor()
        traverser = DataFrameTraverser(processor)
        result = traverser.traverse_and_process(dataframes)
        logger.info("Procesamiento de fechas completado exitosamente")
        return result
    except Exception as e:
        logger.error(f"Error en el procesamiento de fechas: {str(e)}")
        raise


def dropna_all(dataframes: Dict) -> Dict:
    """
    Función wrapper para mantener compatibilidad con el código existente.
    
    Args:
        dataframes: Diccionario con la estructura de datos a limpiar
    
    Returns:
        Dict: Estructura de datos limpia
    """
    try:
        cleaner = DataCleaner()
        
        # Procesar cada DataFrame en la estructura
        for currency_pair, timeframes in dataframes.items():
            for timeframe, dfs in timeframes.items():
                for filename, df in dfs.items():
                    path_info = {
                        'currency_pair': currency_pair,
                        'timeframe': timeframe,
                        'filename': filename
                    }
                    dfs[filename] = cleaner.clean_dataframe(df, path_info)
        
        logger.info("Limpieza de datos completada exitosamente")
        return dataframes
    
    except Exception as e:
        logger.error(f"Error en la limpieza de datos: {str(e)}")
        return dataframes  # Retorna los datos originales en caso de error


class DataFrameValidator:
    """
    Clase para validación de DataFrames siguiendo el principio Single Responsibility.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Inicializa el validador con una configuración específica.
        
        Args:
            config: Configuración de validación opcional
        """
        self.config = config or ValidationConfig()
        self.validation_errors = []
    
    def validate_dataframes(self, dataframes: Dict) -> bool:
        """
        Valida la estructura y tipos de datos de los DataFrames en la jerarquía proporcionada.
        
        Args:
            dataframes: Diccionario jerárquico de DataFrames organizados por par de divisas y timeframe
        
        Returns:
            bool: True si la validación es exitosa, False si hay errores
        
        Raises:
            ValidationError: Si hay errores de validación en modo estricto
        
        Example:
            >>> validator = DataFrameValidator()
            >>> try:
            ...     is_valid = validator.validate_dataframes(dataframes)
            ...     if is_valid:
            ...         print("Validation successful")
            ... except ValidationError as e:
            ...     print(f"Validation failed: {str(e)}")
        """
        try:
            validation_successful = True
            total_files = 0
            error_count = 0
            
            logger.info("Starting DataFrame validation process...")
            
            for currency_pair, timeframes in dataframes.items():
                for timeframe, dfs in timeframes.items():
                    for filename, df in dfs.items():
                        total_files += 1
                        file_context = {
                            'currency_pair': currency_pair,
                            'timeframe': timeframe,
                            'filename': filename
                        }
                        
                        if not self._validate_single_dataframe(df, file_context):
                            validation_successful = False
                            error_count += 1
            
            # Log resultados finales
            if validation_successful:
                logger.info(f"Validation completed successfully. {total_files} files validated.")
            else:
                error_msg = f"Validation completed with {error_count} errors in {total_files} files."
                logger.error(error_msg)
                if self.config.strict_validation:
                    raise ValidationError(error_msg)
            
            return validation_successful
            
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            raise ValidationError(f"Validation process failed: {str(e)}")
    
    def _validate_single_dataframe(self, df: pd.DataFrame, context: Dict[str, str]) -> bool:
        """
        Valida un único DataFrame.
        
        Args:
            df: DataFrame a validar
            context: Información contextual del archivo
        
        Returns:
            bool: True si la validación es exitosa, False si hay errores
        """
        is_valid = True
        
        # Validar número de columnas
        if df.shape[1] != self.config.expected_columns:
            self._log_validation_error(
                f"{context['filename']} under {context['currency_pair']} {context['timeframe']} "
                f"has {df.shape[1]} columns (expected {self.config.expected_columns}).",
                context
            )
            is_valid = False
        
        # Validar tipo de fecha/hora
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[self.config.datetime_column]):
                self._log_validation_error(
                    f"'{self.config.datetime_column}' in {context['filename']} is not of type datetime.",
                    context
                )
                is_valid = False
        except KeyError:
            self._log_validation_error(
                f"Required column '{self.config.datetime_column}' not found in {context['filename']}.",
                context
            )
            is_valid = False
        
        # Validar columnas numéricas
        for col in df.columns:
            if col != self.config.datetime_column:
                if not self._is_numeric_type(df[col]):
                    self._log_validation_error(
                        f"Column '{col}' in {context['filename']} is not of type float or int.",
                        context
                    )
                    is_valid = False
        
        return is_valid
    
    def _is_numeric_type(self, series: pd.Series) -> bool:
        """
        Verifica si una serie es de tipo numérico.
        
        Args:
            series: Serie de pandas a verificar
        
        Returns:
            bool: True si es numérica, False en caso contrario
        """
        return (pd.api.types.is_float_dtype(series) or 
                pd.api.types.is_integer_dtype(series))
    
    def _log_validation_error(self, message: str, context: Dict[str, str]) -> None:
        """
        Registra un error de validación.
        
        Args:
            message: Mensaje de error
            context: Información contextual del error
        """
        error_msg = f"Validation Error in {context['currency_pair']}/{context['timeframe']}: {message}"
        logger.error(error_msg)
        self.validation_errors.append(error_msg)

# Función para exportar el diccionario

def export_data_dict(data_dict, base_dir):
    for pair, timeframes in data_dict.items():
        for timeframe, datasets in timeframes.items():
            for dataset_name, df in datasets.items():
                # Crear la ruta de carpetas
                folder_path = os.path.join(base_dir, pair, timeframe)
                os.makedirs(folder_path, exist_ok=True)
                
                # Ruta del archivo CSV
                file_path = os.path.join(folder_path, f"{dataset_name}.csv")
                
                # Exportar el DataFrame a CSV
                df.to_csv(file_path, index=False)
                print(f"Exportado {dataset_name} a {file_path}")
def main():
    """
    Función principal para ejecutar el flujo completo de procesamiento de datos:
    1. Importación de datos desde archivos CSV.
    2. Estructuración y visualización de la arquitectura de datos.
    3. Corrección de formatos de fecha y hora.
    4. Limpieza de datos, eliminando valores nulos y entradas faltantes.
    5. Validación de la estructura y consistencia de los DataFrames.
    """
    
    # 1. Especifica la ruta de la carpeta que contiene los archivos CSV de datos
    ruta_carpeta = "/home/KAISER/Documents/HermesD_B/HermesD_B/data/raw"
    
    # 2. Importa los archivos CSV en un diccionario de DataFrames organizados por divisa y temporalidad
    DB = import_data(ruta_carpeta)
    
    # 3. Imprime la estructura del diccionario de datos en forma de cascada para una mejor visualización
    imprimir_cascada(DB)
    
    # 4. Arregla el formato de fecha y hora en la columna 'LOCAL TIME' para todos los archivos en el diccionario
    DB = datetime_fix(DB)
    
    # 5. Limpia los datos eliminando filas con valores nulos o incompletos en todos los DataFrames
    DB = dropna_all(DB)
    
    # 6. Valida la estructura y consistencia de los DataFrames tras la limpieza y transformación
    validator = DataFrameValidator()
    try:
        if validator.validate_dataframes(DB):
            print("Data validation completed successfully")
        else:
            print("Data validation completed with errors")
    except ValidationError as e:
        print(f"Validation failed: {str(e)}")
    
    # 7. Exportar  Datos
    # Directorio base donde se guardarán todas las carpetas
    base_dir = "/home/KAISER/Documents/HermesD_B/HermesD_B/data/cleaned" 
    export_data_dict(DB, base_dir)
# Llama a la función principal
if __name__ == "__main__":
    main()