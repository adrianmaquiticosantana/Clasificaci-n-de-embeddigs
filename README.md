Clasificación de Ofertas Laborales usando Word Embeddings y Modelos Supervisados
Este proyecto implementa un sistema de clasificación de descripciones de ofertas laborales en distintas subcategorías dentro del área Tecnologías de la Información - Sistemas, utilizando técnicas de procesamiento de lenguaje natural (NLP) en español y modelos de machine learning supervisados.

Se emplean Word Embeddings para representar las descripciones de texto y se prueban varios clasificadores supervisados con validación cruzada para evaluar su rendimiento.

Estructura del Proyecto
python
Copiar
Editar
Clasificador_Ofertas_TI/
 ├── clasificador_embeddings.py       # Código principal
 ├── CC_Trabajos.csv                  # Dataset de descripciones laborales
 └── SBW-vectors-300-min5.bin.gz      # Modelo preentrenado de Word Embeddings en español
Requisitos
Librerías necesarias:
pandas

numpy

nltk

spacy

gensim

scikit-learn

matplotlib

seaborn

Además, se necesita descargar:

Modelo spaCy en español:

bash
Copiar
Editar
python -m spacy download es_core_news_lg
Stopwords de NLTK:

python
Copiar
Editar
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Descripción del Proceso
Carga de Embeddings:
Se carga un modelo de Word2Vec en español (SBW-vectors-300-min5.bin.gz) utilizando gensim.

Preprocesamiento de texto:

Eliminación de stopwords con spaCy.

Tokenización en oraciones y palabras con NLTK.

Filtrado de tokens alfabéticos y conversión a minúsculas.

Conversión a Embeddings:
Cada documento se representa como el promedio normalizado de los embeddings de sus tokens.

Clasificación:
Se prueban los siguientes clasificadores:

Regresión Logística

Naive Bayes Gaussiano

Bosques Aleatorios

SVM (máquina de vectores de soporte)

Validación Cruzada:
Se realiza validación cruzada estratificada de 5 pliegues para obtener métricas estables.

Evaluación de Desempeño:
Para cada clasificador se reporta:

Accuracy

F1-Score macro

Precision macro

Recall macro

Matriz de confusión global

Reporte de clasificación global detallado

Resultados
Al finalizar, se presentan:

Resumen de métricas promedio y desviaciones estándar por clasificador.

Reportes individuales de desempeño.

Visualización de matrices de confusión.

Se identifica el mejor clasificador según F1-Score macro.

Notas
El dataset debe estar preprocesado y contener al menos las columnas descripción, categoría y subcategoría.

El proyecto está enfocado en descripciones laborales en español.

El modelo de embeddings utilizado debe estar en formato binario compatible con gensim.

Ejemplo de Matriz de Confusión
Se generan gráficos de calor (heatmaps) de las matrices de confusión para cada clasificador utilizando matplotlib y seaborn.

