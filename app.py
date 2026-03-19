import os
import pickle
import streamlit as st

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    padding-top: 1.2rem;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.hero-card, .info-card, .result-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1rem;
    opacity: 0.85;
    margin-bottom: 0;
}
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 16px;
    text-align: center;
}
.small-text {
    font-size: 0.95rem;
    opacity: 0.85;
}
.sample-box {
    background: rgba(255,255,255,0.03);
    border-left: 4px solid #7c3aed;
    padding: 12px 14px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

accuracy_value = None
if os.path.exists("accuracy.txt"):
    with open("accuracy.txt", "r") as f:
        try:
            accuracy_value = float(f.read().strip())
        except:
            accuracy_value = None

st.sidebar.markdown("## ⚙️ Model Details")
st.sidebar.write("**Algorithm:** Multinomial Naive Bayes")
st.sidebar.write("**Text Vectorization:** CountVectorizer")
if accuracy_value is not None:
    st.sidebar.write(f"**Accuracy:** {accuracy_value * 100:.2f}%")
else:
    st.sidebar.write("**Accuracy:** Not available yet")

st.sidebar.markdown("---")
st.sidebar.markdown("## 💡 Sample Messages")
st.sidebar.markdown("**Spam example:**")
st.sidebar.code("Congratulations! You have won a free lottery. Claim your prize now!")
st.sidebar.markdown("**Normal example:**")
st.sidebar.code("Hey, are we still meeting at 6 pm today?")

st.markdown("""
<div class="hero-card">
    <div class="hero-title">📩 Spam Message Classifier</div>
    <p class="hero-subtitle">
        A machine learning web app that classifies text messages as <b>Spam</b> or <b>Not Spam</b>.
        Built with Python, Scikit-learn, and Streamlit.
    </p>
</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-box">
        <h4>🤖 Model</h4>
        <p class="small-text">Naive Bayes Classifier</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-box">
        <h4>🧠 NLP</h4>
        <p class="small-text">Count Vectorization</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    acc_text = f"{accuracy_value * 100:.2f}%" if accuracy_value is not None else "N/A"
    st.markdown(f"""
    <div class="metric-box">
        <h4>📊 Accuracy</h4>
        <p class="small-text">{acc_text}</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.3, 1])

with left:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("✍️ Enter a message")
    msg = st.text_area(
        "Type your message below",
        height=180,
        placeholder="Example: Congratulations! You have won a free reward..."
    )

    c1, c2 = st.columns(2)
    with c1:
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    with c2:
        clear_btn = st.button("🧹 Clear", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    if predict_btn:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("📌 Prediction Result")

        if not msg.strip():
            st.warning("Please enter a message first.")
        else:
            data = cv.transform([msg])
            result = model.predict(data)[0]

            if result == 1:
                st.error("🚫 This message is classified as Spam.")
            else:
                st.success("✅ This message is classified as Not Spam.")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧪 Try these examples")
    st.markdown('<div class="sample-box"><b>Spam:</b><br>Congratulations! You have been selected for a free vacation. Call now!</div>', unsafe_allow_html=True)
    st.markdown('<div class="sample-box"><b>Not Spam:</b><br>Hi mom, I will reach home in 15 minutes.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("📉 Confusion Matrix")
    if os.path.exists("images/confusion_matrix.png"):
        st.image("images/confusion_matrix.png", use_container_width=True)
    else:
        st.info("Confusion matrix image not found. Run model.py first.")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown("---")
st.caption("Developed by Abhishek Jha")