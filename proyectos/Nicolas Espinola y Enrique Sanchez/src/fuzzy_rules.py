import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyRuleSystem:
    def __init__(self):
        # Variables universales
        self.pos_score = ctrl.Antecedent(np.linspace(0, 1, 1000), 'pos_score') # Positividad
        self.neg_score = ctrl.Antecedent(np.linspace(0, 1, 1000), 'neg_score') # Negatividad
        self.sentiment = ctrl.Consequent(np.linspace(0, 10, 1000), 'sentiment') # Sentimiento
        
        # Define funciones de membresía para las variables de entrada
        self.pos_score['low'] = fuzz.trimf(self.pos_score.universe, [0, 0, 0.5]) # Positividad baja
        self.pos_score['medium'] = fuzz.trimf(self.pos_score.universe, [0, 0.5, 1]) # Positividad media
        self.pos_score['high'] = fuzz.trimf(self.pos_score.universe, [0.5, 1, 1]) # Positividad alta
        
        self.neg_score['low'] = fuzz.trimf(self.neg_score.universe, [0, 0, 0.5]) # Negatividad baja
        self.neg_score['medium'] = fuzz.trimf(self.neg_score.universe, [0, 0.5, 1]) # Negatividad media
        self.neg_score['high'] = fuzz.trimf(self.neg_score.universe, [0.5, 1, 1]) # Negatividad alta
        
        # Define funciones de membresía para la variable de salida
        self.sentiment['negative'] = fuzz.trimf(self.sentiment.universe, [0, 0, 5]) # Sentimiento negativo
        self.sentiment['neutral'] = fuzz.trimf(self.sentiment.universe, [0, 5, 10]) # Sentimiento neutral
        self.sentiment['positive'] = fuzz.trimf(self.sentiment.universe, [5, 10, 10]) # Sentimiento positivo
        
        # Define reglas difusas
        self.rules = [
            # R1: Si pos es bajo y neg es bajo entonces el sentimiento es neutral
            ctrl.Rule(self.pos_score['low'] & self.neg_score['low'], 
                     self.sentiment['neutral']),
            
            # R2: Si pos es medio y neg es bajo entonces el sentimiento es positivo
            ctrl.Rule(self.pos_score['medium'] & self.neg_score['low'], 
                     self.sentiment['positive']),
            # R3: Si pos es alto y neg es bajo entonces el sentimiento es positivo
            ctrl.Rule(self.pos_score['high'] & self.neg_score['low'], 
                     self.sentiment['positive']),
            
            # R4: Si pos es bajo y neg es medio entonces el sentimiento es negativo
            ctrl.Rule(self.pos_score['low'] & self.neg_score['medium'], 
                     self.sentiment['negative']),
            
            # R5: Si pos es medio y neg es medio entonces el sentimiento es neutral
            ctrl.Rule(self.pos_score['medium'] & self.neg_score['medium'], 
                     self.sentiment['neutral']),
            
            # R6: Si pos es alto y neg es medio entonces el sentimiento es positivo
            ctrl.Rule(self.pos_score['high'] & self.neg_score['medium'], 
                     self.sentiment['positive']),
            
            # R7: Si pos es bajo y neg es alto entonces el sentimiento es negativo
            ctrl.Rule(self.pos_score['low'] & self.neg_score['high'], 
                     self.sentiment['negative']),
            
            # R8: Si pos es medio y neg es alto entonces el sentimiento es negativo
            ctrl.Rule(self.pos_score['medium'] & self.neg_score['high'], 
                     self.sentiment['negative']),
            
            # R9: Si pos es alto y neg es alto entonces el sentimiento es neutral
            ctrl.Rule(self.pos_score['high'] & self.neg_score['high'], 
                     self.sentiment['neutral'])
        ]
        
        # Crear el sistema de control
        self.sentiment_ctrl = ctrl.ControlSystem(self.rules)
        self.sentiment_simulator = ctrl.ControlSystemSimulation(self.sentiment_ctrl)
    
    def evaluate(self, pos_score, neg_score):
        """
        Evaluar el sistema de control con los valores de entrada
        """
        try:
            self.sentiment_simulator.input['pos_score'] = pos_score
            self.sentiment_simulator.input['neg_score'] = neg_score
            self.sentiment_simulator.compute()
            return self.sentiment_simulator.output['sentiment']
        except:
            # En caso de error, retornar un valor neutral
            return 5.0

def apply_fuzzy_rules(df):
    """
    Aplica las reglas difusas al dataframe con los scores de sentimiento
    """
    fuzzy_system = FuzzyRuleSystem()
    
    # Aplicar reglas difusas a cada fila del dataframe
    results = []
    for _, row in df.iterrows():
        sentiment_score = fuzzy_system.evaluate(
            row['positive_score'], 
            row['negative_score']
        )
        results.append(sentiment_score)
    
    # Agregar columna con los resultados
    df['fuzzy_sentiment'] = results
    return df