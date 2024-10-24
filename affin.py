import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from afinn import Afinn

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Preprocesa el texto siguiendo pasos comunes en NLP:
    1. Convertir a minúsculas
    2. Eliminar URLs
    3. Eliminar menciones (@usuario)
    4. Eliminar hashtags
    5. Eliminar puntuación
    6. Eliminar números
    7. Tokenización
    8. Eliminar stopwords
    9. Lematización
    """
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def get_sentiment_scores_afinn(text):
    """
    Calcula los puntajes de sentimiento usando AFINN
    """
    afinn = Afinn()
    score = afinn.score(text)
    
    # Normalizar el score a valores entre 0 y 1
    # AFINN da valores entre -5 y 5
    pos_score = max(0, score) / 5
    neg_score = abs(min(0, score)) / 5
    
    return pos_score, neg_score

def add_sentiment_scores(df, text_column='sentence'):
    """
    Agrega puntajes de sentimiento positivo y negativo usando AFINN
    """
    sentiment_scores = df[text_column].apply(get_sentiment_scores_afinn)
    
    df['positive_score'] = sentiment_scores.apply(lambda x: x[0])
    df['negative_score'] = sentiment_scores.apply(lambda x: x[1])
    
    return df

def process_dataset(input_file, output_file):
    """
    Procesa el dataset completo
    """
    print("Leyendo el dataset...")
    df = pd.read_csv(input_file)
    
    print("Preprocesando textos...")
    df['processed_text'] = df['sentence'].apply(preprocess_text)
    
    print("Calculando scores de sentimiento...")
    df = add_sentiment_scores(df, 'processed_text')
    
    print("Guardando resultados...")
    df.to_csv(output_file, index=False)
    print("¡Proceso completado!")
    
    return df

if __name__ == "__main__":
    input_file = "./data/test_data.csv"
    output_file = "tweets_processed_afinn.csv"
    
    df_processed = process_dataset(input_file, output_file)