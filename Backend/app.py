from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
from dotenv import load_dotenv
import os

from model_loader import load_transformer_models
from predictor import predict_properties
from utils import build_prompt  
# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Load models once on server start
models = load_transformer_models()

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Backend is up and running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415

    data = request.get_json()
    smiles = data.get("smiles")

    if not smiles:
        return jsonify({"error": "Missing input: 'smiles' is required"}), 400

    try:
        # 1. Predict properties using Transformer
        predictions = predict_properties(smiles, models)

        # 2. Build LLM prompt
        prompt = build_prompt(smiles, predictions)

        # 3. Query DeepSeek (OpenRouter)
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [
                {"role": "system", "content": "You are a scientific research assistant specializing in drug discovery and molecular property analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 700
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(payload))

        if response.status_code != 200:
            return jsonify({"error": "OpenRouter request failed", "details": response.text}), response.status_code

        result = response.json()
        llm_report = result["choices"][0]["message"]["content"]

        return jsonify({
            "smiles": smiles,
            "predictions": predictions,
            "report": llm_report
        })

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
