import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

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

def add_sentiment_scores(df, text_column='sentence'):
    """
    Agrega puntajes de sentimiento positivo y negativo usando VADER
    """
    
    sia = SentimentIntensityAnalyzer()
    
    
    sentiment_scores = df[text_column].apply(lambda x: sia.polarity_scores(x))
    
    # Agregar columnas de sentimiento positivo y negativo
    df['positive_score'] = sentiment_scores.apply(lambda x: x['pos'])
    df['negative_score'] = sentiment_scores.apply(lambda x: x['neg'])
    
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
    df['processed_text'] = df['sentence'].apply(preprocess_text)
    
    # Agregar scores de sentimiento
    print("Calculando scores de sentimiento...")
    df = add_sentiment_scores(df, 'processed_text')
    
    # Guardar resultados
    print("Guardando resultados...")
    df.to_csv(output_file, index=False)
    print("¡Proceso completado!")
    
    return df

# Ejemplo de uso
if __name__ == "__main__":
    input_file = "./data/test_data.csv"
    output_file = "tweets_processed.csv"
    
    df_processed = process_dataset(input_file, output_file)