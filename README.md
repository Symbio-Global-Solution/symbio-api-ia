# 🧠 SYMBIO - API de IA (Python/Flask)

Serviço de Machine Learning para o projeto SYMBIO (Global Solution 2025). Esta API Flask carrega os modelos de IA treinados e os disponibiliza para serem consumidos pela API Java principal.

**Disciplina Relacionada:** *Artificial Intelligence & Chatbot*

## 📦 Entregáveis
Este repositório contém:
* [cite_start]`/api`: O código-fonte da API Flask (`app.py`). [cite: 229]
* [cite_start]`/models`: Os modelos pré-treinados (.pkl / .joblib). [cite: 228]
* [cite_start]`/notebooks`: Os Jupyter Notebooks com o pipeline (Análise, Treino, Avaliação). [cite: 227]
* [cite_start]`/data`: Os datasets (.csv) usados para o treinamento. [cite: 226]

## 🤖 Modelos Implementados
1.  **Classificação de Risco:** Prevê se um cargo tem risco 'ALTO', 'MEDIO' ou 'BAIXO' de automação.
2.  **Clustering de Talentos:** Agrupa colaboradores por perfil comportamental.

## 🛠️ Tecnologias Utilizadas
* Python
* Flask
* Pandas
* Scikit-learn
* Joblib / Pickle

## 🚀 Como Executar (Localmente)

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/](https://github.com/)[seu-usuario]/symbio-api-ia.git
    cd symbio-api-ia
    ```
2.  **Crie e ative um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (Linux/Mac)
    .\venv\Scripts\activate   # (Windows)
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute a API Flask:**
    ```bash
    flask --app api/app run
    ```
5.  A API estará disponível em `http://localhost:5000`.

## 🎛️ Endpoints
* `POST /predict/risk`: Recebe dados do cargo e retorna a classificação.
* `POST /predict/cluster`: Recebe dados do colaborador e retorna o cluster.
