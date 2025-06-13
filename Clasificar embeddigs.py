import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                           recall_score, confusion_matrix, classification_report)
nltk.download('punkt_tab')
nltk.download('stopwords')

# Carga de embeddings
embedding_path = '/content/drive/MyDrive/Colab Notebooks/TESIS/Experimento 2 - Construcción de la BOW y clasificación/SBW-vectors-300-min5.bin.gz'
wv = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

nlp = spacy.load('es_core_news_lg', disable=['parser', 'ner'])

def deleteStopWords(text):
    """Elimina stopwords"""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def word_averaging(wv: KeyedVectors, tokens: list[str]) -> np.ndarray:
    """Promedia embeddings de tokens; si ninguno está en vocab, devuelve cero."""
    vecs = []
    for w in tokens:
        if w in wv.key_to_index:
            vecs.append(wv.get_vector(w))
    if not vecs:
        return np.zeros(wv.vector_size, dtype=np.float32)
    arr = np.vstack(vecs)
    mean = arr.mean(axis=0)
    norm = np.linalg.norm(mean)
    return (mean / norm).astype(np.float32) if norm > 0 else mean.astype(np.float32)

def word_averaging_list(wv: KeyedVectors, token_sequences: list[list[str]]) -> np.ndarray:
    """Aplica word_averaging a cada secuencia y devuelve matriz (n_docs, dim)."""
    return np.array([word_averaging(wv, seq) for seq in token_sequences])

def tokenize_es(text: str) -> list[str]:
    toks = []
    for sent in nltk.sent_tokenize(text, language='spanish'):
        for tok in nltk.word_tokenize(sent, language='spanish'):
            if tok.isalpha() and len(tok) > 1:
                toks.append(tok.lower())
    return toks

# Carga y preprocesamiento de datos
data = pd.read_csv('/content/CC_Trabajos.csv', encoding='utf-8')
data = data[data['categoría'] == 'Tecnologías de la Información - Sistemas']
data = data.dropna(subset=['descripción', 'subcategoría'])

tokenized_texts = (
    data['descripción']
        .map(lambda t: deleteStopWords(t))
        .map(tokenize_es)
        .tolist()
)
labels = data['subcategoría'].tolist()

# Definición de clasificadores
classifiers = {
    'Regresion Logistica': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Bosques Aleatorios': RandomForestClassifier(),
    'SVM': SVC(),
}

# Parámetros de validación cruzada
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Diccionarios para almacenar métricas
accuracy_scores = {name: [] for name in classifiers.keys()}
f1_scores = {name: [] for name in classifiers.keys()}
precision_scores = {name: [] for name in classifiers.keys()}
recall_scores = {name: [] for name in classifiers.keys()}

# Para almacenar todas las predicciones y etiquetas
all_y_test = {name: [] for name in classifiers.keys()}
all_y_pred = {name: [] for name in classifiers.keys()}

# Realizar validación cruzada
print("Realizando validación cruzada...")
fold_count = 1
for train_index, test_index in skf.split(tokenized_texts, labels):
    print(f"Procesando fold {fold_count}/{num_folds}...")
    X_train = [tokenized_texts[i] for i in train_index]
    X_test = [tokenized_texts[i] for i in test_index]
    y_train = [labels[i] for i in train_index]
    y_test = [labels[i] for i in test_index]

    # Embeddings promediados
    X_train_emb = word_averaging_list(wv, X_train)
    X_test_emb = word_averaging_list(wv, X_test)

    # Entrenar y evaluar clasificadores
    for name, clf in classifiers.items():
        clf.fit(X_train_emb, y_train)
        y_pred = clf.predict(X_test_emb)

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        # Guardar métricas
        accuracy_scores[name].append(accuracy)
        f1_scores[name].append(f1)
        precision_scores[name].append(precision)
        recall_scores[name].append(recall)

        # Guardar predicciones y etiquetas reales
        all_y_test[name].extend(y_test)
        all_y_pred[name].extend(y_pred)

    fold_count += 1

# Reporte final
print("\n" + "="*80)
print("REPORTE FINAL DE CLASIFICACIÓN")
print("="*80)

# Crear tablas resumen
summary_accuracy = pd.DataFrame({name: np.mean(accuracy_scores[name]) * 100 for name in classifiers.keys()}, index=['Accuracy (%)']).T
summary_accuracy['Desviación'] = pd.Series({name: np.std(accuracy_scores[name]) * 100 for name in classifiers.keys()})

summary_f1 = pd.DataFrame({name: np.mean(f1_scores[name]) * 100 for name in classifiers.keys()}, index=['F1-Score (%)']).T
summary_f1['Desviación'] = pd.Series({name: np.std(f1_scores[name]) * 100 for name in classifiers.keys()})

summary_precision = pd.DataFrame({name: np.mean(precision_scores[name]) * 100 for name in classifiers.keys()}, index=['Precision (%)']).T
summary_precision['Desviación'] = pd.Series({name: np.std(precision_scores[name]) * 100 for name in classifiers.keys()})

summary_recall = pd.DataFrame({name: np.mean(recall_scores[name]) * 100 for name in classifiers.keys()}, index=['Recall (%)']).T
summary_recall['Desviación'] = pd.Series({name: np.std(recall_scores[name]) * 100 for name in classifiers.keys()})

# Imprimir tablas resumen
print("\n1. MÉTRICAS DE RENDIMIENTO PROMEDIO:")
print("\n1.1 Accuracy:")
print(summary_accuracy.sort_values('Accuracy (%)', ascending=False))
print("\n1.2 F1-Score (Macro):")
print(summary_f1.sort_values('F1-Score (%)', ascending=False))
print("\n1.3 Precision (Macro):")
print(summary_precision.sort_values('Precision (%)', ascending=False))
print("\n1.4 Recall (Macro):")
print(summary_recall.sort_values('Recall (%)', ascending=False))

# Mejor clasificador basado en F1-score
best_classifier = max(classifiers.keys(), key=lambda name: np.mean(f1_scores[name]))
print(f"\nMejor clasificador basado en F1-Score: {best_classifier}")

# Resultados detallados por clasificador
print("\n2. RESULTADOS DETALLADOS POR CLASIFICADOR:")
for idx, name in enumerate(classifiers.keys(), start=1):
    print(f"\n{'-' * 60}")
    print(f"2.{idx} CLASIFICADOR: {name}")
    print(f"{'-' * 60}")

    print(f"Accuracy: {np.mean(accuracy_scores[name]) * 100:.2f}% (±{np.std(accuracy_scores[name]) * 100:.2f}%)")
    print(f"F1-Score (Macro): {np.mean(f1_scores[name]) * 100:.2f}% (±{np.std(f1_scores[name]) * 100:.2f}%)")
    print(f"Precision (Macro): {np.mean(precision_scores[name]) * 100:.2f}% (±{np.std(precision_scores[name]) * 100:.2f}%)")
    print(f"Recall (Macro): {np.mean(recall_scores[name]) * 100:.2f}% (±{np.std(recall_scores[name]) * 100:.2f}%)")

    # Matriz de confusión global
    global_conf_matrix = confusion_matrix(all_y_test[name], all_y_pred[name])
    print("\nMatriz de Confusión Global:")

    classes = sorted(set(all_y_test[name]))
    conf_df = pd.DataFrame(global_conf_matrix, index=classes, columns=classes)
    print("Matriz de Confusión con Subcategorías:")
    print(conf_df.to_string())

    plt.figure(figsize=(10, 8))
    sns.heatmap(global_conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(labels), yticklabels=np.unique(labels))
    plt.title(f"Matriz de Confusión Global para {name}")
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.show()

    # Reporte de clasificación global
    global_class_report = classification_report(all_y_test[name], all_y_pred[name], output_dict=True)
    print("\nReporte de Clasificación Global:")
    print(pd.DataFrame(global_class_report).T)