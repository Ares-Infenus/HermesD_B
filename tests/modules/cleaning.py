import pandas as pd
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuración unificada para todo el procesamiento de datos"""
    # Importación
    allowed_extensions: tuple = ('.csv',)
    encoding: str = 'utf-8'
    
    # Procesamiento de fechas
    date_column: str = 'Local time'
    date_format: str = '%d.%m.%Y %H:%M:%S.%f GMT%z'
    timezone: str = 'UTC'
    
    # Validación
    required_columns: tuple = ('Local time', 'Open', 'High', 'Low', 'Close', 'Volume')
    
    # Exportación
    export_index: bool = False

class DataProcessor:
    """Clase unificada para procesar datos financieros"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    def process_directory(self, input_path: str, output_path: str = None) -> Dict:
        """Procesa todos los archivos CSV en el directorio de entrada"""
        try:
            # Importar y procesar datos
            data_dict = self._import_data(Path(input_path))
            
            # Si se especifica ruta de salida, exportar resultados
            if output_path:
                self._export_data(data_dict, output_path)
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error en el procesamiento: {str(e)}")
            raise
    
    def _import_data(self, directory: Path) -> Dict:
        """Importa recursivamente los archivos CSV del directorio"""
        result = {}
        
        if not directory.exists():
            raise ValueError(f"El directorio {directory} no existe")
        
        for item in directory.rglob("*"):
            if item.is_file() and item.suffix.lower() in self.config.allowed_extensions:
                # Obtener ruta relativa para mantener estructura
                rel_path = item.relative_to(directory)
                parts = list(rel_path.parts[:-1])  # Excluir nombre del archivo
                
                # Procesar el archivo
                df = self._process_file(item)
                
                # Construir estructura de diccionario anidado
                current_dict = result
                for part in parts:
                    current_dict = current_dict.setdefault(part, {})
                current_dict[item.stem] = df
                
                logger.info(f"Procesado: {rel_path}")
        
        return result
    
    def _process_file(self, file_path: Path) -> pd.DataFrame:
        """Procesa un único archivo CSV"""
        try:
            # Leer CSV
            df = pd.read_csv(file_path, encoding=self.config.encoding)
            
            # Validar columnas requeridas
            missing_cols = set(self.config.required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columnas faltantes: {missing_cols}")
            
            # Procesar fechas
            df[self.config.date_column] = pd.to_datetime(
                df[self.config.date_column],
                format=self.config.date_format
            )
            
            if self.config.timezone:
                df[self.config.date_column] = df[self.config.date_column].dt.tz_convert(
                    self.config.timezone
                )
            
            # Limpiar datos
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error procesando {file_path}: {str(e)}")
            raise
    
    def _export_data(self, data_dict: Dict, base_path: str) -> None:
        """Exporta el diccionario de datos procesados"""
        base_path = Path(base_path)
        
        def export_recursive(data: Dict, current_path: Path):
            for key, value in data.items():
                path = current_path / key
                
                if isinstance(value, pd.DataFrame):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value.to_csv(f"{path}.csv", index=self.config.export_index)
                    logger.info(f"Exportado: {path}.csv")
                elif isinstance(value, dict):
                    export_recursive(value, path)
        
        export_recursive(data_dict, base_path)

def main():
    """Función principal"""
    try:
        # Configurar rutas
        input_path = "/home/KAISER/Documents/HermesD_B/HermesD_B/data/raw"
        output_path = "/home/KAISER/Documents/HermesD_B/HermesD_B/data/cleaned"
        
        # Procesar datos
        processor = DataProcessor()
        processed_data = processor.process_directory(input_path, output_path)
        
        logger.info("Procesamiento completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el procesamiento principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()