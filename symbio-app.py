import pickle
import numpy as np
from flask import Flask, request, jsonify

print("--- [SYMBIO API IA] Iniciando servidor ---")

# --- CARREGAMENTO DOS MODELOS ---
# Carregamos os modelos e scalers na memória UMA VEZ.
# Isso garante que a API seja rápida.

# Modelo 1: Classificação
try:
    with open('modelo_risco.pickle', 'rb') as f:
        modelo_classificacao = pickle.load(f)
    print("Modelo 1 (Classificação) carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: 'modelo_risco.pickle' não encontrado.")
    modelo_classificacao = None

# Modelo 2: Agrupamento
try:
    with open('modelo_cluster.pickle', 'rb') as f:
        modelo_cluster = pickle.load(f)
    print("Modelo 2 (Agrupamento) carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: 'modelo_cluster.pickle' não encontrado.")
    modelo_cluster = None

try:
    with open('scaler_cluster.pickle', 'rb') as f:
        scaler_cluster = pickle.load(f)
    print("Scaler 2 (Agrupamento) carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: 'scaler_cluster.pickle' não encontrado.")
    scaler_cluster = None

# --- CRIAÇÃO DO APP FLASK ---
app = Flask(__name__)

# --- ENDPOINT 1: Classificação de Risco ---
@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    """
    Recebe as 3 features do cargo e retorna o risco (ALTO, MEDIO, BAIXO).
    Formato JSON esperado:
    {
        "features": [90, 10, 70]
        (perc_tarefa_repetitiva, perc_exige_criatividade, perc_interacao_humana)
    }
    """
    if not modelo_classificacao:
        return jsonify({"erro": "Modelo de classificação não foi carregado."}), 500

    try:
        data = request.get_json()
        features = data['features']
        
        # Converter para o formato que o modelo espera (array 2D)
        features_np = np.array([features])
        
        # Fazer a predição
        predicao = modelo_classificacao.predict(features_np)
        
        # Retornar o resultado
        return jsonify({"risco_predito": predicao[0]})

    except Exception as e:
        return jsonify({"erro": str(e)}), 400

# --- ENDPOINT 2: Agrupamento de Talentos ---
@app.route('/predict/cluster', methods=['POST'])
def predict_cluster():
    """
    Recebe as 6 features do perfil do funcionário e retorna o cluster.
    Formato JSON esperado:
    {
        "features": [1, 2, 3, 4, 2, 5]
        (JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, 
         PerformanceRating, TrainingTimesLastYear, YearsInCurrentRole)
    }
    """
    if not modelo_cluster or not scaler_cluster:
        return jsonify({"erro": "Modelo de agrupamento não foi carregado."}), 500

    try:
        data = request.get_json()
        features = data['features']
        
        # Converter para array 2D
        features_np = np.array([features])
        
        # !! IMPORTANTE: Aplicar o Scaler
        features_scaled = scaler_cluster.transform(features_np)
        
        # Fazer a predição
        predicao = modelo_cluster.predict(features_scaled)
        
        # Retornar o resultado (converter para int nativo)
        return jsonify({"cluster_predito": int(predicao[0])})

    except Exception as e:
        return jsonify({"erro": str(e)}), 400

# Rota principal (para teste)
@app.route('/')
def home():
    return "API de IA do Projeto SYMBIO está no ar!"

# --- EXECUTAR O SERVIDOR ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)