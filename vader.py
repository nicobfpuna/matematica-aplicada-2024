import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string

import src.fuzzy_rules as fuzzy_rules

# Descargar recursos necesarios de NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")




def preprocess_text(text):
    """
    Preprocesa el texto siguiendo pasos comunes en NLP:
    1. Convertir a minúsculas
    2. Eliminar URLs
    3. Eliminar menciones (@usuario)
    4. Eliminar hashtags
    5. Eliminar puntuación
    6. Eliminar números
    7. Eliminar palabras de una sola letra
    8. Eliminar palabras extendidas (3 o más letras repetidas)
    9. Tokenización
    10. Eliminar stopwords
    11. Lematización
    """

    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"@\w+|#\w+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)
    
    text = re.sub(r"\b\w\b", "", text)
    
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)


def add_sentiment_scores(df, text_column="sentence"):
    """
    Agrega puntajes de sentimiento positivo y negativo usando VADER
    """

    sia = SentimentIntensityAnalyzer()

    sentiment_scores = df[text_column].apply(lambda x: sia.polarity_scores(x))

    # Agregar columnas de sentimiento positivo y negativo
    df["positive_score"] = sentiment_scores.apply(lambda x: x["pos"])
    df["negative_score"] = sentiment_scores.apply(lambda x: x["neg"])

    return df


def process_dataset(input_file, output_file):
    """
    Procesa el dataset completo
    """
    # Leer el dataset
    print("Leyendo el dataset...")
    df = pd.read_csv(input_file)

    # Preprocesar los textos
    print("Preprocesando textos...")
    df["processed_text"] = df["sentence"].apply(preprocess_text)

    # Agregar scores de sentimiento
    print("Calculando scores de sentimiento...")
    df = add_sentiment_scores(df, "processed_text")

    # Guardar resultados
    print("Guardando resultados...")
    df.to_csv(output_file, index=False)
    print("¡Proceso completado!")

    return df


class VaderFuzzifier:
    def __init__(self):
        # Parámetros para la variable de salida (x_op)
        self.output_params = {
            'Negative': {'d': 0, 'e': 0, 'f': 5},
            'Neutral': {'d': 0, 'e': 5, 'f': 10},
            'Positive': {'d': 5, 'e': 10, 'f': 10}
        }
    
    def triangular_membership(self, x, d, e, f):
        """
        Calcula el grado de membresía usando función triangular
        x: valor a evaluar
        d: límite inferior
        e: valor intermedio
        f: límite superior
        """
        if x <= d or x >= f:
            return 0
        elif d < x <= e:
            return (x - d) / (e - d)
        else:  # e < x < f
            return (f - x) / (f - e)
    
    def calculate_range_params(self, scores):
        """
        Calcula los parámetros para las funciones de membresía
        basados en el rango de scores
        """
        min_val = np.min(scores)
        max_val = np.max(scores)
        mid_val = (min_val + max_val) / 2
        
        return {
            'Low': {'d': min_val, 'e': min_val, 'f': mid_val},
            'Medium': {'d': min_val, 'e': mid_val, 'f': max_val},
            'High': {'d': mid_val, 'e': max_val, 'f': max_val}
        }
    
    def fuzzify_score(self, score, params):
        """
        Fuzifica un score individual usando los parámetros dados
        """
        result = {}
        for level, values in params.items():
            membership = self.triangular_membership(
                score, 
                values['d'],
                values['e'],
                values['f']
            )
            result[level] = membership
        return result
    
    def process_vader_scores(self, df, pos_col='positive_score', neg_col='negative_score'):
        """
        Procesa los scores de VADER y retorna los valores fuzificados
        """
        # Calcular parámetros para scores positivos y negativos
        pos_params = self.calculate_range_params(df[pos_col])
        neg_params = self.calculate_range_params(df[neg_col])
        
        # Fuzificar cada score
        results = []
        for _, row in df.iterrows():
            pos_score = row[pos_col]
            neg_score = row[neg_col]
            
            # Fuzificar scores positivos y negativos
            pos_fuzzy = self.fuzzify_score(pos_score, pos_params)
            neg_fuzzy = self.fuzzify_score(neg_score, neg_params)
            
            result = {
                'original_pos': pos_score,
                'original_neg': neg_score,
                'pos_low': pos_fuzzy['Low'],
                'pos_medium': pos_fuzzy['Medium'],
                'pos_high': pos_fuzzy['High'],
                'neg_low': neg_fuzzy['Low'],
                'neg_medium': neg_fuzzy['Medium'],
                'neg_high': neg_fuzzy['High']
            }
            results.append(result)
        
        return pd.DataFrame(results)

# Ejemplo de uso
def fuzzify_vader_results(input_file):
    """
    Función principal para fuzificar resultados de VADER
    """
    # Leer el dataset procesado por VADER
    df = pd.read_csv(input_file)
    
    # Crear instancia del fuzzificador
    fuzzifier = VaderFuzzifier()
    
    # Procesar los scores
    fuzzy_results = fuzzifier.process_vader_scores(df)
    
    # Combinar resultados originales con los fuzificados
    final_df = pd.concat([df, fuzzy_results], axis=1)
    
    return final_df

# Ejemplo de uso
if __name__ == "__main__":
    input_file = "./data/test_data.csv"
    output_file = "tweets_processed.csv"

    df_processed = process_dataset(input_file, output_file)

    fuzzifier = VaderFuzzifier()
    fuzzy_results = fuzzifier.process_vader_scores(df_processed)

    final_df = pd.concat([df_processed, fuzzy_results], axis=1)
    
    output_file = "resultados_fuzificados.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en {output_file}")
    
    #Genera una salida entre 0 y 10, donde valores cerca de cero es negatico
    # Valores cercanos a 5 son neutrales y valores cercanos a 10 son positivos
    print("Generando reglas difusas...")
    fuzzy_df = final_df.copy()
    fuzzy_df = fuzzy_rules.apply_fuzzy_rules(fuzzy_df)
    fuzzy_df.to_csv("data_fuzzy.csv", index=False)
    