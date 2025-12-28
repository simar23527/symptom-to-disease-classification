import streamlit as st
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("Disease Prediction from Symptoms")

text = st.text_area("Describe your symptoms:")

if st.button("Predict"):
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    st.success(pred)
