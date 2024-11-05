import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import argparse
import sys


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



@dataclass
class AppConfig:
    """Configuración global de la aplicación"""
    input_path: Path
    log_file: Optional[Path] = None
    verbose: bool = False
    show_structure: bool = True

class Interface:
    """Interfaz principal de la aplicación"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @staticmethod
    def _configurar_argumentos() -> argparse.Namespace:
        """Configura y parsea los argumentos de línea de comandos"""
        parser = argparse.ArgumentParser(
            description='Importador de archivos CSV con estructura jerárquica'
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
                log_file=Path(args.log) if args.log else None,
                verbose=args.verbose,
                show_structure=not args.no_structure
            )
            
            # Configurar logging
            interface._configurar_logging(config)
            interface.logger.info(f"Iniciando importación desde {config.input_path}")
            
            # Configuración del importador
            import_config = ImportConfig(
                encoding='utf-8',
                separator=',',
                max_file_size=50 * 1024 * 1024,  # 50MB
                skip_hidden_files=True
            )

            # Crear y ejecutar importador
            importador = ImportadorCSV(config.input_path, import_config)
            datos = importador.importar()

            # Mostrar resultados
            stats = importador.obtener_estadisticas()
            interface._mostrar_estadisticas(stats)
            
            # Mostrar estructura si está habilitado
            if config.show_structure:
                print("\n=== Estructura de Archivos ===")
                imprimir_estructura(datos)
            
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