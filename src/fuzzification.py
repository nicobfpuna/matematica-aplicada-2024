import numpy as np
import pandas as pd

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