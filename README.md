# HermesData

## Descripción
**HermesDB** es una base de datos robusta diseñada para almacenar y gestionar datos históricos de candlestick a diversas temporalidades, facilitando un análisis detallado de activos financieros. Esta plataforma optimizada permite la captura de información de precios en diferentes intervalos de tiempo, desde 1 minuto hasta 1 mes, proporcionando una vista integral del comportamiento del mercado.

### Características principales:
- **Temporalidades soportadas**: 
  - 1 Mes
  - 1 Semana
  - 1 Día
  - 4 Horas
  - 1 Hora

- **Activos disponibles**: 
  - EUR/USD
  - GBP/USD
  - USD/CAD
  - USD/JPY
  - AUD/USD
  - USD/CHF
  - XAG/USD
  - XAU/USD
  - BTC/USD

- **Tipos de precio**: 
  - **Bid**: El precio al que se puede vender un activo.
  - **Ask**: El precio al que se puede comprar un activo.
  - **Mid**: El promedio entre el precio Bid y el Ask.

A través de **HermesDB**, analistas y traders podrán acceder a patrones históricos, tendencias y movimientos de precios, permitiendo decisiones informadas en sus estrategias de inversión. La base de datos está optimizada para consultas rápidas, asegurando que los usuarios puedan extraer y analizar datos sin demoras. Además, incluye herramientas para la visualización de datos y análisis técnico, facilitando la identificación de patrones y oportunidades en el mercado.

## Instalación
(Todavía por aclarar. Se proporcionará información detallada sobre los requisitos del sistema y las instrucciones de instalación en futuras versiones.)

## Uso
Se prevé que **HermesDB** sea utilizada para optimizar búsquedas y maximizar el rendimiento en análisis de datos financieros. Los usuarios podrán aplicar técnicas de análisis técnico y desarrollar estrategias de inversión efectivas.

## Contribuciones
(Todavía no hay contribuciones. Las contribuciones serán bienvenidas una vez que el producto esté más avanzado.)

## Licencia
Este proyecto se desarrollará como código abierto para que todos puedan utilizarlo y contribuir. Se busca proporcionar una solución accesible para la búsqueda de datos históricos de manera efectiva.

Estado |=                 |10%
Buscando Archivo

Estructura de carpetas

/mi_proyecto
│
├── /data
│   ├── /raw               # Datos originales sin procesar (CSV sucios)
│   ├── /cleaned           # Datos limpios después del procesamiento
│   └── /processed         # Datos procesados (si aplica)
│
├── /notebooks             # Jupyter notebooks para análisis y limpieza exploratoria
│
├── /scr               # Scripts de Python para limpieza y transformación de datos
│   ├── data_cleaning.py   # Script principal de limpieza de datos
│   ├── data_transformation.py  # Script para transformar datos si es necesario
│   └── utils.py           # Funciones utilitarias que se pueden reutilizar
│
├── /config                # Archivos de configuración
│   ├── config.yaml        # Configuración del proyecto (rutas, parámetros, etc.)
│   └── database_config.py  # Configuración específica para la conexión a Oracle DB
│
├── /db                    # Scripts y archivos relacionados con la base de datos
│   ├── create_schema.sql   # Script SQL para crear la estructura de la base de datos
│   ├── insert_data.sql      # Script SQL para insertar datos en la base de datos
│   └── queries.sql          # Consultas SQL útiles para la base de datos
│
├── /tests                  # Pruebas automatizadas
│   ├── test_data_cleaning.py  # Pruebas para el script de limpieza de datos
│   └── test_database.py        # Pruebas para interacciones con la base de datos
│
├── /logs                   # Archivos de registro de procesos
│   └── process_log.log      # Registro de la limpieza de datos y errores
│
└── README.md               # Documentación del proyecto


Registro de Actualizaciones del Proyecto

    0.1: Creación de funciones de limpieza de datos.
    0.2: Implementación de cálculo y recorte de datos.
    0.3: Migración de datos desde archivos CSV a Oracle SQL.
    0.4: Desarrollo de funciones de verificación y manejo de errores para mejorar la legibilidad del código.
    0.5: Carga del 60% de los datos.
    0.8: Carga del 70% de los datos.
    1.3: Carga del 80% de los datos.
    2.1: Carga del 90% de los datos.
    3.4: Carga completa del 100% de los datos.
    5.5: Creación de documentación en formato .docx.
    8.9: Generación de documentación en formato .txt.
    14.4: Desarrollo del módulo de instalación del proyecto.
    23.3: Creación del módulo de licencia de software.

Lanzamiento de la Versión Final

    1.0.0: Redacción profesional del README y finalización del proyecto.

Nuevas Características en Desarrollo

    2.0: Reducción de datos a intervalos de 30, 15, 5 y 1 minuto.
    3.0: Incorporación de precios para las 20 principales acciones del mercado.