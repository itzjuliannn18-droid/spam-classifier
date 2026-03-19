import streamlit as st
import pickle
import os

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

# Title + Description
st.title("📩 Spam Message Classifier")
st.markdown("""
This app uses **Machine Learning (Naive Bayes)** to classify messages as **Spam or Not Spam**.

👉 Enter a message below or try a sample.
""")

# Sidebar (extra pro touch)
st.sidebar.header("📊 Model Info")
st.sidebar.write("Algorithm: Naive Bayes")
st.sidebar.write("Vectorizer: CountVectorizer")
st.sidebar.write("Accuracy: ~98%")  # You can replace with your actual accuracy

# Sample messages
st.subheader("💡 Try Sample Messages")

sample1 = "Congratulations! You have won a free lottery. Claim now!"
sample2 = "Hey bro, are we meeting today?"

col1, col2 = st.columns(2)

if col1.button("📨 Try Spam Example"):
    st.session_state.message = sample1

if col2.button("📨 Try Normal Example"):
    st.session_state.message = sample2

# Input box
msg = st.text_area("✍️ Enter your message", value=st.session_state.get("message", ""))

# Prediction
if st.button("🔍 Predict"):
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        data = cv.transform([msg])
        result = model.predict(data)[0]

        st.subheader("📌 Result")

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam")

# Show Confusion Matrix Image
st.subheader("📊 Model Performance")

if os.path.exists("images/confusion_matrix.png"):
    st.image("images/confusion_matrix.png", caption="Confusion Matrix")
else:
    st.info("Run model.py to generate confusion matrix image.")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by Abhishek Jha")
