import pandas as pd
from flask import Flask
import joblib

# Initialize App
app = Flask(__name__)

# Load models
model = joblib.load('model/model_binary.dat.gz')

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction

def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, model)
    if prediction == 1:
        label = "M"
    else:
        label = "B"
    return {
        'status': 200,
        'label': label,
        'prediction': int(prediction)
    }
