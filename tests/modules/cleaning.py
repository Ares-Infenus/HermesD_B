import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List,Union
from dataclasses import dataclass
from enum import Enum
import argparse
import sys
from datetime import datetime
from enum import Enum
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ImportError(Exception):
    """Clase base para excepciones de importación"""
    pass

class FileNotFoundError(ImportError):
    """Error cuando no se encuentra el archivo"""
    pass

class InvalidCSVError(ImportError):
    """Error cuando el CSV es inválido"""
    pass

class PermissionError(ImportError):
    """Error de permisos"""
    pass

@dataclass
class ImportConfig:
    """Configuración para la importación"""
    encoding: str = 'utf-8'
    separator: str = ','
    max_file_size: int = 1024 * 1024 * 100  # 100MB
    supported_extensions: tuple = ('.csv',)
    skip_hidden_files: bool = True
    chunk_size: int = None

class ImportadorCSV:
    """Clase para importar archivos CSV de forma recursiva"""

    def __init__(self, carpeta: str, config: ImportConfig = None):
        """
        Inicializa el importador
        
        Args:
            carpeta: Ruta a la carpeta a procesar
            config: Configuración de importación
        """
        self.carpeta = Path(carpeta)
        self.config = config or ImportConfig()
        self.datos: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validar_carpeta()

    def _validar_carpeta(self) -> None:
        """Valida que la carpeta exista y sea accesible"""
        if not self.carpeta.exists():
            raise FileNotFoundError(f"La carpeta {self.carpeta} no existe")
        if not self.carpeta.is_dir():
            raise NotADirectoryError(f"{self.carpeta} no es un directorio")
        if not os.access(self.carpeta, os.R_OK):
            raise PermissionError(f"No hay permisos de lectura para {self.carpeta}")

    def _validar_archivo(self, ruta: Path) -> None:
        """
        Valida un archivo antes de procesarlo
        
        Args:
            ruta: Ruta al archivo a validar
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            PermissionError: Si no hay permisos de lectura
            ValueError: Si el archivo es demasiado grande
        """
        if not ruta.exists():
            raise FileNotFoundError(f"El archivo {ruta} no existe")
        if not os.access(ruta, os.R_OK):
            raise PermissionError(f"No hay permisos de lectura para {ruta}")
        if ruta.stat().st_size > self.config.max_file_size:
            raise ValueError(f"El archivo {ruta} excede el tamaño máximo permitido")

    def _es_archivo_valido(self, archivo: Path) -> bool:
        """
        Verifica si un archivo debe ser procesado
        
        Args:
            archivo: Ruta al archivo a verificar
            
        Returns:
            bool: True si el archivo debe procesarse
        """
        if self.config.skip_hidden_files and archivo.name.startswith('.'):
            return False
        return archivo.suffix.lower() in self.config.supported_extensions

    def _leer_csv(self, ruta: Path) -> pd.DataFrame:
        """
        Lee un archivo CSV
        
        Args:
            ruta: Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con los datos del CSV
            
        Raises:
            InvalidCSVError: Si hay errores al leer el CSV
        """
        try:
            return pd.read_csv(
                ruta,
                encoding=self.config.encoding,
                sep=self.config.separator,
                chunksize=self.config.chunk_size
            )
        except pd.errors.EmptyDataError:
            raise InvalidCSVError(f"El archivo {ruta} está vacío")
        except pd.errors.ParserError:
            raise InvalidCSVError(f"Error al parsear {ruta}. Verifique el formato")
        except UnicodeDecodeError:
            raise InvalidCSVError(f"Error de codificación en {ruta}. Intente con otro encoding")
        except Exception as e:
            raise InvalidCSVError(f"Error desconocido al leer {ruta}: {str(e)}")

    def importar(self) -> Dict[str, Any]:
        """
        Importa todos los CSV de forma recursiva
        
        Returns:
            Dict: Estructura de datos con los DataFrames importados
            
        Raises:
            ImportError: Si hay errores durante la importación
        """
        self.logger.info(f"Iniciando importación desde {self.carpeta}")
        try:
            self.datos = self._importar_csvs()
            self.logger.info("Importación completada exitosamente")
            return self.datos
        except Exception as e:
            self.logger.error(f"Error durante la importación: {str(e)}")
            raise

    def _importar_csvs(self) -> Dict[str, Any]:
        """
        Proceso principal de importación
        
        Returns:
            Dict: Estructura de datos con los DataFrames importados
        """
        datos = {}
        archivos_procesados = 0
        errores = 0

        for ruta in self.carpeta.rglob('*'):
            if not ruta.is_file() or not self._es_archivo_valido(ruta):
                continue

            try:
                self._validar_archivo(ruta)
                ruta_relativa = str(ruta.parent.relative_to(self.carpeta))
                nombre_archivo = ruta.stem

                # Crear estructura de directorios si no existe
                if ruta_relativa not in datos:
                    datos[ruta_relativa] = {}

                # Leer y almacenar el CSV
                self.logger.info(f"Procesando {ruta}")
                datos[ruta_relativa][nombre_archivo] = self._leer_csv(ruta)
                archivos_procesados += 1

            except ImportError as e:
                self.logger.error(f"Error al procesar {ruta}: {str(e)}")
                errores += 1
                continue

        self.logger.info(f"Procesados {archivos_procesados} archivos con {errores} errores")
        return datos

    def obtener_estadisticas(self) -> Dict[str, Any]:
        """
        Retorna estadísticas sobre los datos importados
        
        Returns:
            Dict: Estadísticas de la importación
        """
        stats = {
            'total_archivos': 0,
            'total_registros': 0,
            'archivos_por_carpeta': {},
            'memoria_utilizada': 0
        }

        for carpeta, archivos in self.datos.items():
            stats['archivos_por_carpeta'][carpeta] = len(archivos)
            stats['total_archivos'] += len(archivos)
            
            for df in archivos.values():
                stats['total_registros'] += len(df)
                stats['memoria_utilizada'] += df.memory_usage(deep=True).sum()

        return stats
    
class ItemType(Enum):
    FOLDER = "📁"
    FILE = "📄"

@dataclass
class FormatConfig:
    indent_size: int = 4
    max_depth: int = 10

def format_line(name: str, level: int, item_type: ItemType, config: FormatConfig) -> str:
    """Formatea una línea con la indentación y el ícono apropiado."""
    indent = ' ' * level * config.indent_size
    return f"{indent}{item_type.value} {name}"

def imprimir_estructura(
    diccionario: Dict[str, Any], 
    nivel: int = 0, 
    config: FormatConfig = FormatConfig()
) -> None:
    """
    Imprime la estructura de un diccionario de forma jerárquica.
    
    Args:
        diccionario: Diccionario a imprimir
        nivel: Nivel actual de profundidad
        config: Configuración de formato
        
    Raises:
        RecursionError: Si se excede la profundidad máxima
    """
    try:
        if nivel > config.max_depth:
            raise RecursionError("Profundidad máxima excedida")
            
        for clave, valor in diccionario.items():
            # Imprime la clave actual
            print(format_line(clave, nivel, ItemType.FOLDER, config))
            
            # Procesa el valor si es un diccionario
            if isinstance(valor, dict):
                imprimir_estructura(valor, nivel + 1, config)
            else:
                print(format_line(clave, nivel + 1, ItemType.FILE, config))
                
    except (TypeError, AttributeError) as e:
        raise ValueError(f"Diccionario inválido: {e}")

#########################################################################################################################

class DataFrameProcessor:
    """Clase para procesar DataFrames"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _try_datetime_formats(self, series: pd.Series) -> pd.Series:
        """
        Intenta diferentes formatos de fecha hasta encontrar uno que funcione.
        
        Args:
            series: Serie de pandas con las fechas en formato string
            
        Returns:
            pd.Series: Serie con las fechas convertidas a datetime
        """
        # Lista de formatos a probar
        formatos = [
        '%d.%m.%Y %H:%M:%S.%f GMT%z',    # Formato con milisegundos y zona horaria
        '%d.%m.%Y %H:%M:%S GMT%z',       # Formato sin milisegundos pero con zona horaria
        '%d.%m.%Y %H:%M:%S.%f',          # Sin zona horaria
        '%d.%m.%Y %H:%M:%S',             # Sin zona horaria ni milisegundos
        '%Y-%m-%d %H:%M:%S.%f GMT%z',    # Formato alternativo con zona horaria
        '%Y-%m-%d %H:%M:%S GMT%z',       # Formato alternativo sin milisegundos
        '%d.%m.%Y %H:%M:%S.%f',
        '%d.%m.%Y %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%d.%m.%Y %H:%M:%S.%f %Z',
        '%d.%m.%Y %H:%M:%S.%f %z',
        '%Y-%m-%d %H:%M:%S %Z',
        '%Y-%m-%d %H:%M:%S %z',
        '%Y-%m-%d %H:%M:%S.%f %z',  # Con microsegundos y offset
        '%Y-%m-%d %H:%M:%S %Z'      # Con nombre de zona horaria
        ]
        
        for formato in formatos:
            try:
                return pd.to_datetime(series, format=formato)
            except ValueError:
                continue
                
        # Si ningún formato funciona, intentar el parsing automático
        try:
            return pd.to_datetime(series)
        except Exception as e:
            self.logger.error(f"No se pudo convertir la fecha: {str(e)}")
            return pd.to_datetime(series, errors='coerce')

    def convertir_local_time(self, 
                           dataframes: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Convierte la columna 'Local time' de los DataFrames a tipo datetime.
        
        Args:
            dataframes: Diccionario de DataFrames importados
            
        Returns:
            Dict: Diccionario con los DataFrames procesados
        """
        processed_data = {}
        conversion_stats = {
            'procesados': 0,
            'exitosos': 0,
            'fallidos': 0,
            'errores': []
        }
        
        for carpeta, archivos in dataframes.items():
            processed_data[carpeta] = {}
            
            for nombre_archivo, df in archivos.items():
                conversion_stats['procesados'] += 1
                df_procesado = df.copy()
                
                if 'Local time' in df_procesado.columns:
                    try:
                        # Intentar conversión con múltiples formatos
                        df_procesado['Local time'] = self._try_datetime_formats(df_procesado['Local time'])
                        
                        # Verificar valores NaT
                        nat_count = df_procesado['Local time'].isna().sum()
                        total_rows = len(df_procesado)
                        
                        if nat_count > 0:
                            self.logger.warning(
                                f"{nat_count}/{total_rows} valores no convertidos en {carpeta}/{nombre_archivo}"
                            )
                            
                        if nat_count < total_rows:  # Si al menos algunos valores se convirtieron
                            conversion_stats['exitosos'] += 1
                            self.logger.info(
                                f"Convertida columna 'Local time' en {carpeta}/{nombre_archivo}"
                                f" ({total_rows - nat_count}/{total_rows} valores válidos)"
                            )
                        else:
                            conversion_stats['fallidos'] += 1
                            error_msg = f"Ningún valor pudo ser convertido en {carpeta}/{nombre_archivo}"
                            conversion_stats['errores'].append(error_msg)
                            self.logger.error(error_msg)
                            
                    except Exception as e:
                        conversion_stats['fallidos'] += 1
                        error_msg = f"Error en {carpeta}/{nombre_archivo}: {str(e)}"
                        conversion_stats['errores'].append(error_msg)
                        self.logger.error(error_msg)
                        
                processed_data[carpeta][nombre_archivo] = df_procesado
                
        # Registrar estadísticas finales
        self.logger.info(f"""
            Procesamiento completado:
            - Total procesados: {conversion_stats['procesados']}
            - Exitosos: {conversion_stats['exitosos']}
            - Fallidos: {conversion_stats['fallidos']}
        """)
        
        if conversion_stats['errores']:
            self.logger.warning("Errores encontrados durante la conversión:")
            for error in conversion_stats['errores'][:5]:  # Mostrar solo los primeros 5 errores
                self.logger.warning(error)
            if len(conversion_stats['errores']) > 5:
                self.logger.warning(f"... y {len(conversion_stats['errores']) - 5} errores más")
                
        return processed_data

######################################################################################################################
class ValidationErrorType(Enum):
    """Tipos de errores de validación"""
    NOT_DATAFRAME = "NOT_DATAFRAME"
    INVALID_COLUMNS = "INVALID_COLUMNS"
    INVALID_DATETIME = "INVALID_DATETIME"
    INVALID_NUMERIC = "INVALID_NUMERIC"

@dataclass
class ValidationError:
    """Clase para almacenar información detallada de errores"""
    error_type: ValidationErrorType
    pair: str
    details: str
    column: Optional[str] = None

class DataFrameValidator:
    """Clase responsable de la validación de DataFrames siguiendo el principio SRP"""
    
    def __init__(self, required_columns: List[str]):
        """
        Inicializa el validador con las columnas requeridas
        
        Args:
            required_columns: Lista de columnas que debe tener cada DataFrame
        """
        self.required_columns = required_columns
        self.errors: List[ValidationError] = []

    def validate_instance(self, df: pd.DataFrame, pair: str) -> bool:
        """Valida que el objeto sea un DataFrame"""
        if not isinstance(df, pd.DataFrame):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.NOT_DATAFRAME,
                    pair,
                    f"{pair} no es un DataFrame"
                )
            )
            return False
        return True

    def validate_columns(self, df: pd.DataFrame, pair: str) -> bool:
        """Valida que el DataFrame tenga las columnas correctas"""
        if list(df.columns) != self.required_columns:
            self.errors.append(
                ValidationError(
                    ValidationErrorType.INVALID_COLUMNS,
                    pair,
                    f"Columnas encontradas: {list(df.columns)}",
                )
            )
            return False
        return True

    def validate_datetime(self, df: pd.DataFrame, pair: str, datetime_col: str) -> bool:
        """Valida que la columna datetime tenga el tipo correcto"""
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            self.errors.append(
                ValidationError(
                    ValidationErrorType.INVALID_DATETIME,
                    pair,
                    f"La columna {datetime_col} no es de tipo datetime",
                    datetime_col
                )
            )
            return False
        return True

    def validate_numeric_columns(self, df: pd.DataFrame, pair: str, numeric_columns: List[str]) -> bool:
        """Valida que las columnas numéricas tengan el tipo correcto"""
        is_valid = True
        for col in numeric_columns:
            if not pd.api.types.is_float_dtype(df[col]):
                self.errors.append(
                    ValidationError(
                        ValidationErrorType.INVALID_NUMERIC,
                        pair,
                        f"La columna {col} no es de tipo float",
                        col
                    )
                )
                is_valid = False
        return is_valid

class DataFrameValidatorService:
    """Clase de servicio que coordina la validación (Facade Pattern)"""
    
    def __init__(self):
        self.required_columns = ['Local time', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.validator = DataFrameValidator(self.required_columns)
        
    def validate_dataframes(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> List[ValidationError]:
        """
        Valida la estructura de los DataFrames
        
        Args:
            data_dict: Diccionario con estructura {timeframe: {pair: DataFrame}}
        
        Returns:
            List[ValidationError]: Lista de errores encontrados
        """
        numeric_columns = [col for col in self.required_columns if col != 'Local time']
        
        for time_frame, pairs in data_dict.items():
            for pair, df in pairs.items():
                # Aplicamos validaciones en cadena
                if (self.validator.validate_instance(df, pair) and
                    self.validator.validate_columns(df, pair)):
                    self.validator.validate_datetime(df, pair, 'Local time')
                    self.validator.validate_numeric_columns(df, pair, numeric_columns)
        
        return self.validator.errors

def format_validation_errors(errors: List[ValidationError]) -> str:
    """
    Formatea los errores de validación para su presentación
    
    Args:
        errors: Lista de errores de validación
    
    Returns:
        str: Mensaje formateado con los errores
    """
    if not errors:
        return "Validación completada exitosamente. No se encontraron errores."
    
    error_messages = ["Se encontraron los siguientes errores:"]
    for error in errors:
        message = f"- Par {error.pair}: {error.details}"
        if error.column:
            message += f" (Columna: {error.column})"
        error_messages.append(message)
    
    return "\n".join(error_messages)

# Ejemplo de uso
def validate_and_report(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """
    Función principal que ejecuta la validación y reporta los resultados
    
    Args:
        data_dict: Diccionario con estructura {timeframe: {pair: DataFrame}}
    """
    try:
        validator_service = DataFrameValidatorService()
        errors = validator_service.validate_dataframes(data_dict)
        print(format_validation_errors(errors))
    except Exception as e:
        print(f"Error inesperado durante la validación: {str(e)}")
###################################################################################################################
class ExportError(Exception):
    """Base class for export-related exceptions"""
    pass

class FileSystemError(ExportError):
    """Raised when there are filesystem-related issues"""
    pass

class DataFrameError(ExportError):
    """Raised when there are DataFrame-related issues"""
    pass

@dataclass
class ExportMetrics:
    """Tracks metrics during the export process"""
    total_files: int = 0
    successful_exports: int = 0
    failed_exports: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        """Returns the total duration of the export operation"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def success_rate(self) -> float:
        """Returns the percentage of successful exports"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_exports / self.total_files) * 100

class ExportFormat(Enum):
    """Supported export formats"""
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "xlsx"

@dataclass
class ExportConfig:
    """Configuration for the export process"""
    format: ExportFormat = ExportFormat.CSV
    chunk_size: Optional[int] = None
    compression: Optional[str] = None
    encoding: str = 'utf-8'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    max_workers: int = 4
    buffer_size: int = 8192  # 8KB buffer for file operations
    verify_exports: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds

class DataDictionaryExporter:
    """
    A robust exporter for hierarchical data dictionaries containing DataFrames.
    
    This class implements NASA's reliability principles and SOLID design patterns:
    - Single Responsibility: Focuses solely on exporting data
    - Open/Closed: Easily extensible for new export formats
    - Liskov Substitution: Uses proper inheritance for error handling
    - Interface Segregation: Clean, focused interface
    - Dependency Inversion: Configurable through ExportConfig
    
    Key features:
    - Robust error handling with retries
    - Concurrent exports for performance
    - Progress tracking and metrics
    - Data verification
    - Resource cleanup
    """
    
    def __init__(self, config: ExportConfig):
        """
        Initialize the exporter with configuration
        
        Args:
            config: ExportConfig object containing export parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = ExportMetrics()
        
    def _verify_dataframe_export(self, df: pd.DataFrame, export_path: Path) -> bool:
        """
        Verifies that a DataFrame was correctly exported
        
        Args:
            df: Original DataFrame
            export_path: Path to exported file
            
        Returns:
            bool: True if verification passes
        """
        try:
            if self.config.format == ExportFormat.CSV:
                df_verify = pd.read_csv(export_path, encoding=self.config.encoding)
            elif self.config.format == ExportFormat.PARQUET:
                df_verify = pd.read_parquet(export_path)
            elif self.config.format == ExportFormat.EXCEL:
                df_verify = pd.read_excel(export_path)
                
            # Verify row count and column names match
            if len(df) != len(df_verify) or not all(df.columns == df_verify.columns):
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Verification failed for {export_path}: {str(e)}")
            return False

    def _export_dataframe(self, df: pd.DataFrame, path: Path, retries: int = 0) -> bool:
        """
        Exports a single DataFrame with retry logic
        
        Args:
            df: DataFrame to export
            path: Export path
            retries: Current retry count
            
        Returns:
            bool: True if export was successful
            
        Raises:
            DataFrameError: If export fails after all retries
        """
        try:
            if self.config.format == ExportFormat.CSV:
                df.to_csv(
                    path,
                    index=False,
                    encoding=self.config.encoding,
                    date_format=self.config.date_format,
                    chunksize=self.config.chunk_size
                )
            elif self.config.format == ExportFormat.PARQUET:
                df.to_parquet(
                    path,
                    compression=self.config.compression
                )
            elif self.config.format == ExportFormat.EXCEL:
                df.to_excel(
                    path,
                    index=False,
                    date_format=self.config.date_format
                )
                
            if self.config.verify_exports and not self._verify_dataframe_export(df, path):
                raise DataFrameError("Export verification failed")
                
            return True
            
        except Exception as e:
            if retries < self.config.max_retries:
                self.logger.warning(f"Export attempt {retries + 1} failed for {path}: {str(e)}")
                time.sleep(self.config.retry_delay)
                return self._export_dataframe(df, path, retries + 1)
            raise DataFrameError(f"Failed to export {path} after {retries} retries: {str(e)}")

    def _process_dict_item(self, value: Any, current_path: Path, key: str) -> None:
        """
        Processes a single dictionary item for export
        
        Args:
            value: The value to process (DataFrame or dict)
            current_path: Current directory path
            key: Dictionary key being processed
        """
        try:
            if isinstance(value, pd.DataFrame):
                export_path = current_path / f"{key}.{self.config.format.value}"
                if self._export_dataframe(value, export_path):
                    self.metrics.successful_exports += 1
                    self.metrics.total_bytes += export_path.stat().st_size
            elif isinstance(value, dict):
                new_path = current_path / key
                self._recursive_export(value, new_path)
            else:
                self.logger.warning(
                    f"Skipping '{key}': not a DataFrame or dict. "
                    f"Type: {type(value)}"
                )
                
        except Exception as e:
            self.metrics.failed_exports += 1
            self.logger.error(f"Error processing {key}: {str(e)}")
            raise

    def _recursive_export(self, current_dict: Dict, current_path: Path) -> None:
        """
        Recursively exports dictionary contents
        
        Args:
            current_dict: Dictionary to process
            current_path: Current directory path
        """
        try:
            current_path.mkdir(parents=True, exist_ok=True)
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_dict_item, value, current_path, key
                    ): key 
                    for key, value in current_dict.items()
                }
                
                for future in as_completed(futures):
                    key = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Failed to process {key}: {str(e)}")
                        
        except Exception as e:
            raise FileSystemError(f"Error accessing directory {current_path}: {str(e)}")

    def export(self, data_dict: Dict, base_dir: Union[str, Path]) -> ExportMetrics:
        """
        Exports a hierarchical dictionary of DataFrames to files
        
        Args:
            data_dict: Dictionary containing DataFrames
            base_dir: Base directory for export
            
        Returns:
            ExportMetrics: Export operation metrics
            
        Raises:
            ExportError: If export operation fails
        """
        try:
            base_path = Path(base_dir)
            self.metrics = ExportMetrics()
            self.metrics.start_time = time.time()
            
            # Clean up existing directory if needed
            if base_path.exists():
                shutil.rmtree(base_path)
                
            self._recursive_export(data_dict, base_path)
            
            self.metrics.end_time = time.time()
            self.logger.info(self._format_metrics())
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise ExportError(f"Export operation failed: {str(e)}")
        
    def _format_metrics(self) -> str:
        """Formats metrics for logging"""
        return f"""
        Export completed:
        - Duration: {self.metrics.duration:.2f} seconds
        - Files: {self.metrics.total_files}
        - Successful: {self.metrics.successful_exports}
        - Failed: {self.metrics.failed_exports}
        - Success rate: {self.metrics.success_rate:.1f}%
        - Total size: {self.metrics.total_bytes / 1024 / 1024:.2f} MB
        """

# Integration with Interface class
def add_exporter_to_interface(interface_class: type) -> type:
    """
    Adds DataDictionaryExporter functionality to the Interface class
    
    Args:
        interface_class: The Interface class to modify
        
    Returns:
        type: Modified Interface class
    """
    original_init = interface_class.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.dict_exporter = DataDictionaryExporter(
            ExportConfig(
                format=ExportFormat.CSV,
                max_workers=4,
                verify_exports=True
            )
        )
    
    def export_data_dict(self, data_dict: Dict, output_dir: Union[str, Path]) -> ExportMetrics:
        """
        Exports data dictionary using the enhanced exporter
        
        Args:
            data_dict: Dictionary of DataFrames to export
            output_dir: Output directory
            
        Returns:
            ExportMetrics: Export operation metrics
        """
        try:
            return self.dict_exporter.export(data_dict, output_dir)
        except ExportError as e:
            self.logger.error(f"Export failed: {str(e)}")
            raise
    
    interface_class.__init__ = new_init
    interface_class.export_data_dict = export_data_dict
    return interface_class
###################################################################################################################
@dataclass
class AppConfig:
    """Configuración global de la aplicación"""
    input_path: Path
    output_path: Path = Path('/home/KAISER/Documents/HermesD_B/HermesD_B/data/cleaned')
    log_file: Optional[Path] = None
    verbose: bool = False
    show_structure: bool = True
    datetime_format: str = '%d.%m.%Y %H:%M:%S.%f %Z%z'
    export_format: ExportFormat = ExportFormat.CSV
    export_chunk_size: Optional[int] = None
    export_compression: Optional[str] = None
    export_max_workers: int = 4
    export_verify: bool = True

class Interface:
    """Interfaz principal de la aplicación"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processor = DataFrameProcessor()

        # Inicializar el exportador con configuración por defecto
        self.exporter = DataDictionaryExporter(
            ExportConfig(
                format=ExportFormat.CSV,
                max_workers=4,
                verify_exports=True,
                chunk_size=None,
                compression=None,
                encoding='utf-8',
                date_format='%Y-%m-%d %H:%M:%S'
            )
        )

    @staticmethod
    def _configurar_argumentos() -> argparse.Namespace:
        """Configura y parsea los argumentos de línea de comandos"""
        parser = argparse.ArgumentParser(
            description='Importador y exportador de archivos con estructura jerárquica'
        )
        parser.add_argument(
            '--output',
            type=str,
            default="/home/KAISER/Documents/HermesD_B/HermesD_B/data/cleaned",
            help='Ruta de salida para los archivos exportados'
        )
        parser.add_argument(
            '--export-format',
            type=str,
            choices=['csv', 'parquet', 'xlsx'],
            default='csv',
            help='Formato de exportación (csv, parquet, xlsx)'
        )
        parser.add_argument(
            '--export-workers',
            type=int,
            default=4,
            help='Número de workers para exportación paralela'
        )
        parser.add_argument(
            '--export-chunk-size',
            type=int,
            help='Tamaño de chunk para exportación (opcional)'
        )
        parser.add_argument(
            '--no-verify-exports',
            action='store_true',
            help='Deshabilitar verificación de exportaciones'
        )
        parser.add_argument(
            '--path', 
            type=str,
            default="/home/KAISER/Documents/HermesD_B/HermesD_B/data/raw",
            help='Ruta a la carpeta con los archivos CSV'
        )
        parser.add_argument(
            '--log', 
            type=str,
            help='Ruta al archivo de log'
        )
        parser.add_argument(
            '--verbose', 
            action='store_true',
            help='Mostrar información detallada'
        )
        parser.add_argument(
            '--no-structure', 
            action='store_true',
            help='No mostrar la estructura de archivos'
        )
        parser.add_argument(
            '--datetime-format',
            type=str,
            default='%d.%m.%Y %H:%M:%S.%f %Z%z',
            help='Formato de fecha y hora para la conversión'
        )       
        return parser.parse_args()

    def _configurar_logging(self, config: AppConfig) -> None:
        """Configura el sistema de logging"""
        log_handlers = []
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        log_handlers.append(console_handler)
        
        # Handler para archivo si se especificó
        if config.log_file:
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            log_handlers.append(file_handler)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.DEBUG if config.verbose else logging.INFO,
            handlers=log_handlers
        )

    def _mostrar_estadisticas(self, stats: dict) -> None:
        """Muestra las estadísticas de importación de forma formateada"""
        print("\n=== Estadísticas de Importación ===")
        print(f"📊 Total archivos: {stats['total_archivos']}")
        print(f"📝 Total registros: {stats['total_registros']}")
        print(f"💾 Memoria utilizada: {stats['memoria_utilizada'] / 1024 / 1024:.2f} MB")
        
        print("\n📂 Archivos por carpeta:")
        for carpeta, cantidad in stats['archivos_por_carpeta'].items():
            print(f"  {carpeta or '.'}: {cantidad} archivos")
    
    
    def _procesar_datos(self, datos: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Procesa los datos importados aplicando las transformaciones necesarias"""
        try:
            self.logger.info("Iniciando procesamiento de datos...")

            # Convertir columnas de tiempo
            datos_procesados = self.processor.convertir_local_time(datos)
            
            # Validador de estructura
            validador = validate_and_report(datos_procesados)

            self.logger.info("Procesamiento de datos completado")
            return datos_procesados
            
        except Exception as e:
            self.logger.error(f"Error durante el procesamiento de datos: {str(e)}")
            raise
    
    def _configurar_exportador(self, config: AppConfig) -> None:
        """Configura el exportador con los parámetros especificados"""
        export_config = ExportConfig(
            format=config.export_format,
            chunk_size=config.export_chunk_size,
            compression=config.export_compression,
            max_workers=config.export_max_workers,
            verify_exports=config.export_verify
        )
        self.exporter = DataDictionaryExporter(export_config)

    def _exportar_datos(self, datos: Dict[str, Dict[str, pd.DataFrame]], output_path: Path) -> None:
        """
        Exporta los datos procesados al formato especificado
        
        Args:
            datos: Diccionario de DataFrames a exportar
            output_path: Ruta de salida para los archivos
        """
        try:
            self.logger.info(f"Iniciando exportación de datos a {output_path}")
            
            metrics = self.exporter.export(datos, output_path)
            
            self.logger.info("=== Estadísticas de Exportación ===")
            self.logger.info(f"Duración: {metrics.duration:.2f} segundos")
            self.logger.info(f"Archivos procesados: {metrics.total_files}")
            self.logger.info(f"Exportaciones exitosas: {metrics.successful_exports}")
            self.logger.info(f"Exportaciones fallidas: {metrics.failed_exports}")
            self.logger.info(f"Tasa de éxito: {metrics.success_rate:.1f}%")
            self.logger.info(f"Tamaño total: {metrics.total_bytes / 1024 / 1024:.2f} MB")
            
        except ExportError as e:
            self.logger.error(f"Error durante la exportación: {str(e)}")
            raise

    @staticmethod
    def main() -> int:
        """
        Punto de entrada principal de la aplicación
        
        Returns:
            int: Código de salida (0 para éxito, otro valor para error)
        """
        interface = Interface()
        
        try:
            # Parsear argumentos
            args = interface._configurar_argumentos()
            
            # Crear configuración de la aplicación
            config = AppConfig(
                input_path=Path(args.path),
                output_path=Path(args.output),
                log_file=Path(args.log) if args.log else None,
                verbose=args.verbose,
                show_structure=not args.no_structure,
                export_format=ExportFormat(args.export_format),
                export_chunk_size=args.export_chunk_size,
                export_max_workers=args.export_workers,
                export_verify=not args.no_verify_exports
            )
            
            # Configurar logging
            interface._configurar_logging(config)
            interface._configurar_exportador(config)

            interface.logger.info(f"Iniciando importación desde {config.input_path}")
            
            # Configuración del importador
            import_config = ImportConfig(
                encoding='utf-8',
                separator=',',
                max_file_size=50 * 1024 * 1024,  # 50MB
                skip_hidden_files=True
            )

            # Importar y procesar datos
            importador = ImportadorCSV(config.input_path, import_config)
            datos = importador.importar()
            datos_procesados = interface._procesar_datos(datos)

            # Mostrar resultados
            stats = importador.obtener_estadisticas()
            interface._mostrar_estadisticas(stats)
            
            # Mostrar estructura si está habilitado
            if config.show_structure:
                print("\n=== Estructura de Archivos ===")
                imprimir_estructura(datos_procesados)
            
            # Exportar datos
            interface._exportar_datos(datos_procesados, config.output_path)

            interface.logger.info("Proceso completado exitosamente")
            return 0

        except ImportError as e:
            interface.logger.error(f"Error de importación: {e}")
            return 1
        except KeyboardInterrupt:
            interface.logger.info("Proceso interrumpido por el usuario")
            return 130
        except Exception as e:
            interface.logger.exception(f"Error inesperado: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(Interface.main())