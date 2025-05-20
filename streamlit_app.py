import streamlit as st
import joblib

# Load model
model = joblib.load("model/spam_classifier.pkl")

st.title("📱 Spam SMS Classifier")
st.write("Enter an SMS message to check if it's spam or not.")

message = st.text_area("Enter your message:")

if st.button("Classify"):
    result = model.predict([message])
    label = "🚫 Spam" if result[0] == 1 else "✅ Not Spam"
    st.success(f"Prediction: {label}")