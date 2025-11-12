import pickle
import numpy as np
from flask import Flask, request, jsonify

print(f"*--- [SYMBIO] Iniciando servidor ---*\n")

# Carregamento dos Modelos
try:
    with open('modelo_risco.pickle', 'rb') as f:
        modelo_classificacao = pickle.load(f)
    print("Modelo 1 - Classificação carregado.")

    with open('label_encoder_risco.pickle', 'rb') as f:
        label_encoder_classificacao = pickle.load(f)
    print("LabelEncoder 1 - Classificação carregado.")

    with open('modelo_cluster.pickle', 'rb') as f:
        modelo_cluster = pickle.load(f)
    print("Modelo 2 - Agrupamento carregado.")

    with open('scaler_cluster.pickle', 'rb') as f:
        scaler_cluster = pickle.load(f)
    print("Scaler 2 - Agrupamento carregado.")
    
    print(f"*--- Modelos carregados com sucesso ---*\n\n")

except FileNotFoundError as e:
    print(f"[ERRO:]: Arquivo de modelo não encontrado: {e.filename}")
    modelo_classificacao = None
    label_encoder_classificacao = None
    modelo_cluster = None
    scaler_cluster = None
except Exception as e:
    print(f"[ERRO:] Ao carregar modelos: {e}")

# Aplicação Flask
app = Flask(__name__)

# ENDPOINT 1: Classificação de Risco 
@app.route('/prever/risco', methods=['POST'])
def prever_risco():
    """
    Recebe as 3 features do cargo e retorna o risco (ALTO, MEDIO, BAIXO).
    """
    if not modelo_classificacao or not label_encoder_classificacao:
        return jsonify({"erro": "Modelo de classification não foi carregado no servidor."}), 500

    try:
        data = request.get_json(force=True)
        if 'features' not in data or len(data['features']) != 3:
            return jsonify({"erro": "JSON deve ter a chave 'features' com uma lista de 3 números."}), 400
            
        features = data['features']
        features_np = np.array([features])
        
        # Fazer a predição
        predicao_numerica = modelo_classificacao.predict(features_np)
        
        # Traduzir a predição de volta para texto
        predicao_texto = label_encoder_classificacao.inverse_transform(predicao_numerica)
        
        # Retornar o resultado em texto
        return jsonify({"risco_predito": predicao_texto[0]})

    except Exception as e:
        print(f"Erro no endpoint /prever/risco: {e}")
        return jsonify({"erro": f"Erro interno do servidor: {e}"}), 500

# ENDPOINT 2: Agrupamento de Talentos 
@app.route('/prever/cluster', methods=['POST'])
def prever_cluster():
    """
    Recebe as 6 features do perfil do funcionário e retorna o cluster.
    """
    if not modelo_cluster or not scaler_cluster:
        return jsonify({"erro": "Modelo de agrupamento não foi carregado no servidor."}), 500

    try:
        data = request.get_json(force=True)
        if 'features' not in data or len(data['features']) != 6:
             return jsonify({"erro": "JSON deve ter a chave 'features' com uma lista de 6 números."}), 400
             
        features = data['features']
        features_np = np.array([features])
        
        # Aplicar o Scaler
        features_scaled = scaler_cluster.transform(features_np)
        
        # Fazer a predição
        predicao = modelo_cluster.predict(features_scaled)
        
        # Retornar o resultado
        return jsonify({"cluster_predito": int(predicao[0])})

    except Exception as e:
        print(f"Erro no endpoint /predict/cluster: {e}")
        return jsonify({"erro": f"Erro interno do servidor: {e}"}), 500

# Rota Principal
@app.route('/')
def home():
    return "SYMBIO API de IA está no ar!"

# Executar Programa
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)