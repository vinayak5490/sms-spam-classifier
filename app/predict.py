# app/predict.py

import joblib
import os

# Load the model
model_path = os.path.join("model", "spam_classifier.pkl")
model = joblib.load(model_path)

def predict_spam(message: str) -> str:
    prediction = model.predict([message])
    return "Spam" if prediction[0] == 1 else "Not Spam"
