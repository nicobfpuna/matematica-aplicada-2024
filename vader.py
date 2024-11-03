import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
import time

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
    df["positive_score"] = sentiment_scores.apply(lambda x: x["pos"])
    df["negative_score"] = sentiment_scores.apply(lambda x: x["neg"])
    return df

def process_dataset(input_file, output_file):
    """
    Procesa el dataset completo
    """
    print("Leyendo el dataset...")
    df = pd.read_csv(input_file)
    print("Preprocesando textos...")
    df["processed_text"] = df["sentence"].apply(preprocess_text)
    print("Calculando scores de sentimiento...")
    df = add_sentiment_scores(df, "processed_text")
    print("Guardando resultados...")
    df.to_csv(output_file, index=False)
    print("¡Proceso completado!")
    return df

class VaderFuzzifier:
    def __init__(self):
        self.output_params = {
            'Negative': {'d': 0, 'e': 0, 'f': 5},
            'Neutral': {'d': 0, 'e': 5, 'f': 10},
            'Positive': {'d': 5, 'e': 10, 'f': 10}
        }
    
    def triangular_membership(self, x, d, e, f):
        if x <= d or x >= f:
            return 0
        elif d < x <= e:
            return (x - d) / (e - d)
        else:
            return (f - x) / (f - e)
    
    def calculate_range_params(self, scores):
        min_val = np.min(scores)
        max_val = np.max(scores)
        mid_val = (min_val + max_val) / 2
        return {
            'Low': {'d': min_val, 'e': min_val, 'f': mid_val},
            'Medium': {'d': min_val, 'e': mid_val, 'f': max_val},
            'High': {'d': mid_val, 'e': max_val, 'f': max_val}
        }
    
    def fuzzify_score(self, score, params):
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
        pos_params = self.calculate_range_params(df[pos_col])
        neg_params = self.calculate_range_params(df[neg_col])
        results = []
        for _, row in df.iterrows():
            start_time = time.time()
            pos_score = row[pos_col]
            neg_score = row[neg_col]
            pos_fuzzy = self.fuzzify_score(pos_score, pos_params)
            neg_fuzzy = self.fuzzify_score(neg_score, neg_params)
            fuzzification_time = time.time() - start_time
            result = {
                'original_pos': pos_score,
                'original_neg': neg_score,
                'pos_low': pos_fuzzy['Low'],
                'pos_medium': pos_fuzzy['Medium'],
                'pos_high': pos_fuzzy['High'],
                'neg_low': neg_fuzzy['Low'],
                'neg_medium': neg_fuzzy['Medium'],
                'neg_high': neg_fuzzy['High'],
                'fuzzification_time': fuzzification_time
            }
            results.append(result)
        return pd.DataFrame(results)

def triangular_mf(x, d, e, f):
    if x <= d or x >= f:
        return 0
    elif d < x <= e:
        return (x - d) / (e - d)
    else:
        return (f - x) / (f - e)

def defuzzify_centroid(row, output_range=(0, 10), num_samples=1000):
    z = np.linspace(output_range[0], output_range[1], num_samples)
    output_mfs = {
        'Negative': {'d': 0, 'e': 0, 'f': 5},
        'Neutral': {'d': 0, 'e': 5, 'f': 10},
        'Positive': {'d': 5, 'e': 10, 'f': 10}
    }
    numerator = 0
    denominator = 0
    for zi in z:
        negative_mf = triangular_mf(zi, **output_mfs['Negative']) * max(row['neg_low'], row['neg_medium'], row['neg_high'])
        neutral_mf = triangular_mf(zi, **output_mfs['Neutral']) * row['pos_medium']
        positive_mf = triangular_mf(zi, **output_mfs['Positive']) * max(row['pos_medium'], row['pos_high'])
        max_membership = max(negative_mf, neutral_mf, positive_mf)
        numerator += zi * max_membership
        denominator += max_membership
    if denominator == 0:
        return 5.0
    return numerator / denominator

def process_fuzzified_data(df):
    def defuzzify_row(row):
        start_time = time.time()
        defuzzified_value = defuzzify_centroid(row)
        defuzzification_time = time.time() - start_time
        return defuzzified_value, defuzzification_time
    
    defuzzified_values = []
    defuzzification_times = []
    for _, row in df.iterrows():
        defuzzified_value, defuzzification_time = defuzzify_row(row)
        defuzzified_values.append(defuzzified_value)
        defuzzification_times.append(defuzzification_time)
    
    df['defuzzified_value'] = defuzzified_values
    df['defuzzification_time'] = defuzzification_times
    df['final_sentiment'] = df['defuzzified_value'].apply(lambda x: 'Negative' if x <= 3.3 else 'Neutral' if x <= 6.7 else 'Positive')
    return df

if __name__ == "__main__":
    ## Creación de dataset procesado
    input_file = "./data/test_data.csv"
    output_file = "tweets_processed.csv"
    ## Procesamiento de dataset con VADER
    df_processed = process_dataset(input_file, output_file)
    ## Fuzzificación de los resultados de VADER
    fuzzifier = VaderFuzzifier()
    fuzzy_results = fuzzifier.process_vader_scores(df_processed)
    final_df = pd.concat([df_processed, fuzzy_results], axis=1)
    output_file = "resultados_fuzificados.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en {output_file}")
    
    fuzzified_df = pd.read_csv("resultados_fuzificados.csv")
    ## Defuzzificación de los resultados
    final_df = process_fuzzified_data(fuzzified_df)
    final_df['total_processing_time'] = final_df['fuzzification_time'] + final_df['defuzzification_time']
    output_file = "resultados_defuzzificados.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Resultados guardados en {output_file}")
    print(final_df.head())
    print("Proceso de defuzzificación completado.")
