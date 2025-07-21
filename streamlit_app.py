import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os

# ----- Hide menu/branding -----
st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="centered")
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        div.stButton > button {
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

# ----- NLTK Prep -----
@st.cache_resource
def download_nltk_data():
    downloads = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for item in downloads:
        try:
            nltk.download(item, quiet=True)
        except Exception:
            pass
download_nltk_data()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()
    words = [w for w in words if w.isalpha() and w not in stop_words]
    try:
        words = [lemmatizer.lemmatize(w) for w in words]
    except Exception:
        pass
    return ' '.join(words)

# ----- Load Models -----
@st.cache_resource
def load_models():
    try:
        lgr = joblib.load("lg_fake_news_model.pkl") 
        nb = joblib.load("nb_fake_news_model.pkl")
        vec = joblib.load("tfidf_vectorizer.pkl")
        return lgr, nb, vec
    except Exception:
        return None, None, None
lgr_model, nb_model, vectorizer = load_models()

def predict_news(text, model_choice):
    if not text.strip():
        return None, "❗ Please enter news text."
    if not lgr_model or not nb_model or not vectorizer:
        return None, "❗ Models not available."
    clean = preprocess(text)
    if not clean: return None, "❗ No meaningful words after preprocessing."
    vec = vectorizer.transform([clean])
    model = nb_model if model_choice == "Naive Bayes" else lgr_model
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    if pred == 1:
        verdict = "🟢 **True News**"
        color = "success"
    else:
        verdict = "🔴 **Fake News**"
        color = "error"
    details = (
        f"{verdict}\n\n"
        f"**Confidence**: `{proba[pred]*100:.2f}%`\n\n"
        f"**Probability:**\n"
        f"- Fake: {proba[0]*100:.2f}%\n"
        f"- True: {proba[1]*100:.2f}%"
    )
    return (color, details), ""

# ---- Sidebar: EXAMPLES ----
st.sidebar.markdown("## ✨ Example Articles")
st.sidebar.info("Select an example below to fill the input box.", icon="🧩")
example_texts = {
    "Real News Example": "WASHINGTON (Reuters) - The U.S. Supreme Court on Monday declined to hear a challenge to a lower court ruling that upheld the Affordable Care Act's individual mandate requiring Americans to obtain health insurance.",
    "Fake News Example": "BREAKING: Scientists have discovered that drinking coffee every morning will make you live forever! This incredible breakthrough was hidden by big pharma for decades!",
}
# Maintain proper widget state
if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""
for name, text in example_texts.items():
    if st.sidebar.button(name):
        st.session_state["article_text"] = text

# ---- HEADER ----
st.markdown("<h2 style='text-align:center'>🔍 Fake News Detector</h2>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-size:1.06rem;color:gray;'>
Paste news below, choose a model, then Analyze.<br>
Or choose an example from the sidebar.
</div>
""", unsafe_allow_html=True)
st.markdown("")

# ---- MAIN FORM ----
with st.form("predict_form", clear_on_submit=False):
    article_text = st.text_area(
        "Paste your news article below:",
        height=180,
        placeholder="Paste news article here...",
        key="article_text"
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        model_choice = st.selectbox("Prediction model:",
            ["Logistic Regression", "Naive Bayes"],
            key="model_choice")
    with col2:
        submit = st.form_submit_button("🔎 Analyze!", use_container_width=True)
    msg = ""
    if submit:
        with st.spinner("Analyzing..."):
            result, msg = predict_news(article_text, model_choice)
        if result:
            color, details = result
            if color == "success":
                st.success(details)
            else:
                st.error(details)
        elif msg:
            st.warning(msg)

