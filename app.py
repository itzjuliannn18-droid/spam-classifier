import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Message Classifier")

msg = st.text_area("Enter your message")

if st.button("Predict"):
    data = cv.transform([msg])
    result = model.predict(data)[0]

    if result == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Not Spam")