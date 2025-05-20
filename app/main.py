# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_spam

app = FastAPI()

class Message(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Spam SMS Classifier API is running"}

@app.post("/predict")
def classify_sms(msg: Message):
    result = predict_spam(msg.text)
    return {"prediction": result}
