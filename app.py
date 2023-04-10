import datetime

from flask import request
import pandas as pd

from functions import app
from functions import get_model_response

model_name = "Breast Cancer Wisconsin (Diagnostic)"
model_file = 'model_binary.dat.gz'
version = "v1.0.0"

@app.route('/info', methods=['GET'])
def info():
    return {
        "name": model_name,
        "version": version,
        "timestamp": datetime.datetime.now()
    }

@app.route('/health', methods=['GET'])
def health():
    return {"status": "OK"}

@app.route('/predict', methods=['POST'])
def predict():
    features = request.get_json()
    if not features:
        return {"error": "No features provided"}, 500
    try:
        response = get_model_response(features)
    except ValueError as e:
        return {"error": str(e)}, 500
    
    return response, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0')