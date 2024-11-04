## README

### Descripción del Script

Este script de Python fue hecho especificamente para la materia de Matematica Aplicada de la FPUNA - procesa y clasifica datos de texto (específicamente tweets) en categorías de sentimiento. Utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y lógica difusa para asignar etiquetas de sentimiento (Negativo, Neutro, Positivo) a cada tweet. La clasificación se realiza mediante el análisis de sentimientos de VADER y una fuzzificación de los puntajes obtenidos. Los datos procesados se exportan a un archivo CSV para su análisis posterior.

### Requisitos

1. **Python**: Asegúrese de tener Python 3.7 o superior instalado.
2. **Entorno virtual (`venv`)**: Se recomienda el uso de un entorno virtual para gestionar las dependencias.

### Configuración del Entorno Virtual y Dependencias

1. **Crear un entorno virtual**:

   ```bash
   python -m venv venv
   ```

2. **Activar el entorno virtual**:
   - En Windows:

     ```bash
     venv\Scripts\activate
     ```

   - En macOS y Linux:

     ```bash
     source venv/bin/activate
     ```

3. **Instalar dependencias**:

   Asegúrese de tener un archivo `requirements.txt` con las dependencias necesarias. Ejecute el siguiente comando para instalar las bibliotecas requeridas:

   ```bash
   pip install -r requirements.txt
   ```

### Contenido del Archivo `requirements.txt`

El archivo `requirements.txt` debe incluir las siguientes dependencias:

Aquí tienes la tabla de dependencias y sus versiones:

| Paquete               | Versión       |
|-----------------------|---------------|
| afinn                 | 0.1           |
| certifi               | 2024.8.30     |
| charset-normalizer    | 3.4.0         |
| click                 | 8.1.7         |
| idna                  | 3.10          |
| joblib                | 1.4.2         |
| networkx              | 3.4.2         |
| nltk                  | 3.9.1         |
| numpy                 | 2.1.2         |
| packaging             | 24.1          |
| pandas                | 2.2.3         |
| python-dateutil       | 2.9.0.post0   |
| pytz                  | 2024.2        |
| regex                 | 2024.9.11     |
| requests              | 2.32.3        |
| scikit-fuzzy          | 0.5.0         |
| scipy                 | 1.14.1        |
| six                   | 1.16.0        |
| tqdm                  | 4.66.5        |
| tzdata                | 2024.2        |
| urllib3               | 2.2.3         |
| vaderSentiment        | 3.3.2         |

Esta tabla facilita la visualización de las dependencias junto con sus respectivas versiones.

### Descarga de Recursos de NLTK

El script descarga ciertos recursos necesarios de NLTK para el análisis de texto. Asegúrese de estar conectado a internet la primera vez que ejecute el script:

- `punkt` para tokenización de palabras.
- `stopwords` para eliminar palabras vacías.
- `wordnet` para lematización.
- `vader_lexicon` para el análisis de sentimientos.

### Uso

1. **Archivos de Entrada y Salida**:
   - `INPUT_FILE_TEST`: archivo CSV con datos de prueba.
   - `OUTPUT_FILE_TEST`: archivo CSV donde se guardarán los resultados procesados de prueba.
   - `INPUT_FILE`: archivo CSV con datos de entrenamiento.
   - `OUTPUT_FILE`: archivo CSV donde se guardarán los resultados procesados.

2. **Ejecución del Script**:
   Con el entorno virtual activado, ejecute el script en un entorno de Python:

   ```bash
   python vader.py
   ```

3. **Pipeline de Procesamiento**:
   - **Modulo 1**: Preprocesamiento del texto (limpieza, tokenización, eliminación de palabras vacías, lematización).
   - **Modulo 2**: Análisis de sentimiento con VADER, que genera puntajes positivos y negativos.
   - **Modulo 3**: Fuzzificación de los puntajes de sentimiento mediante lógica difusa.
   - **Modulo 4**: Aplicación de reglas difusas usando el módulo `fuzzy_rules`.
   - **Modulo 5**: Defuzzificación de los puntajes de sentimiento difusos para obtener una clasificación final.
   - **Modulo 6**: Estadísticas del procesamiento, incluyendo tiempos y distribución de categorías de sentimiento.

4. **Almacenamiento de Resultados**:
   Los datos procesados se guardan en un archivo CSV especificado en la variable `OUTPUT_FILE`.

### Funciones Principales

1. **`preprocess_text(text)`**: Realiza la limpieza y preprocesamiento del texto.
2. **`add_sentiment_scores(df, text_column="sentence")`**: Genera puntajes de sentimiento con VADER.
3. **`process_dataset(input_file)`**: Lee el archivo CSV y aplica el preprocesamiento de texto y análisis de sentimiento.
4. **`VaderFuzzifier`**: Clase que realiza la fuzzificación de los puntajes de sentimiento.
5. **`defuzzify_centroid(row, output_range=(0, 10), num_samples=1000)`**: Calcula el valor defuzzificado del sentimiento.
6. **`process_fuzzified_data(df)`**: Defuzzifica los datos fuzzificados y asigna una categoría de sentimiento final.
7. **`print_statistics(df)`**: Muestra estadísticas de procesamiento y tiempos.

### Ejemplo de Estructura del Archivo de Entrada

El archivo de entrada debe tener una columna llamada `sentence` que contenga el texto de cada tweet.

### Ejemplo de Estructura del Archivo de Salida

El archivo de salida tendrá las siguientes columnas adicionales:

- `processed_text`: texto procesado.
- `positive_score` y `negative_score`: puntajes de sentimiento calculados con VADER.
- `pos_low`, `pos_medium`, `pos_high`: puntajes fuzzificados para el sentimiento positivo.
- `neg_low`, `neg_medium`, `neg_high`: puntajes fuzzificados para el sentimiento negativo.
- `defuzzified_value`: valor defuzzificado para el sentimiento.
- `final_sentiment`: clasificación final (Negativo, Neutro, Positivo).
- `fuzzification_time` y `defuzzification_time`: tiempos de procesamiento.
- `total_processing_time`: tiempo total de procesamiento para cada tweet.

### Notas

- Asegúrese de que el módulo `src.fuzzy_rules` esté implementado correctamente y sea accesible en el path del script.
- El archivo de entrada debe estar en el formato CSV y contener una columna llamada `sentence`.

### Alumnos Responsables

- **Enrique Sanchez**  
  GitHub: [xHenrySx](https://github.com/xHenrySx)

- **Nicolas Espinola**  
  GitHub: [nicobfpuna](https://github.com/nicobfpuna)