import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import re
import string

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('sentiwordnet')

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

def penn_to_wn(tag):
    """
    Convierte etiquetas Penn Treebank a formato WordNet
    """
    if tag.startswith('J'):
        return 'a'  # adjetivo
    elif tag.startswith('V'):
        return 'v'  # verbo
    elif tag.startswith('N'):
        return 'n'  # sustantivo
    elif tag.startswith('R'):
        return 'r'  # adverbio
    else:
        return None
def get_sentiment_scores_sentiwordnet(text):
    """
    Calcula los puntajes de sentimiento usando SentiWordNet
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    pos_scores = []
    neg_scores = []
    
    for word, tag in tagged:
        wn_tag = penn_to_wn(tag)
        if wn_tag not in ('a', 'v', 'n', 'r'):
            continue
        
        synsets = list(swn.senti_synsets(word, wn_tag))
        if not synsets:
            continue
            
        # Promedio de scores para todos los synsets
        pos = sum(s.pos_score() for s in synsets) / len(synsets)
        neg = sum(s.neg_score() for s in synsets) / len(synsets)
        
        # Normalizar los puntajes
        total = pos + neg
        if total > 0:
            pos /= total
            neg /= total
        
        pos_scores.append(pos)
        neg_scores.append(neg)
    
    if not pos_scores:  # Si no se encontraron palabras con sentimiento
        return 0.0, 0.0
        
    return sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores)

def add_sentiment_scores(df, text_column='sentence'):
    """
    Agrega puntajes de sentimiento positivo y negativo usando SentiWordNet
    """
    sentiment_scores = df[text_column].apply(get_sentiment_scores_sentiwordnet)
    
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
    output_file = "tweets_processed_sentiwordnet.csv"
    
    df_processed = process_dataset(input_file, output_file)