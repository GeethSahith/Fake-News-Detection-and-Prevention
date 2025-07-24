import gradio as gr
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import numpy as np
import os

def download_nltk_data():
    """Downloads NLTK resources if not already present."""
    required_data = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for item in required_data:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            print(f"Downloading NLTK data: {item}...")
            nltk.download(item, quiet=True)

download_nltk_data()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Cleans and preprocesses raw text for model prediction."""
    if not isinstance(text, str):
        return ""
    # Lowercase and remove punctuation/digits
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(lemmatized_words)

def load_models():
    """Loads the pre-trained models and vectorizer from disk."""
    try:
        lgr = joblib.load("lgr_fake_news_model.pkl")
        nb = joblib.load("nb_fake_news_model.pkl")
        vec = joblib.load("tfidf_vectorizer.pkl")
        print("Models loaded successfully.")
        return lgr, nb, vec
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}. Please ensure model files are in the correct directory.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred while loading models: {e}")
        return None, None, None

lgr_model, nb_model, vectorizer = load_models()

def predict_news(article_text, model_choice):
    """
    Analyzes the article text using the selected model and returns the prediction,
    confidence, and a probability plot.
    """
    # 1. Input Validation
    if not article_text or not article_text.strip():
        return "Please enter a news article to analyze.", None
    if lgr_model is None or nb_model is None or vectorizer is None:
        return "Error: A required model file is missing. The application cannot make predictions.", None

    # 2. Preprocess Text
    cleaned_text = preprocess_text(article_text)
    if not cleaned_text.strip():
        return "After preprocessing, no meaningful text remains. Please provide a more substantial article.", None

    # 3. Vectorize and Predict
    vector = vectorizer.transform([cleaned_text])
    model = nb_model if model_choice == "Naive Bayes" else lgr_model
    
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    # 4. Format Output
    is_true_news = (prediction == 1)
    label = "✅ Real News" if is_true_news else "❌ Fake News"
    confidence = probabilities[prediction] * 100
    
    result_color = "#34D399" if is_true_news else "#F87171" # Green for real, Red for fake

    output_message = f"""
    <div class='result-container'>
        <div class='result-label' style='color: {result_color};'>
            {label}
        </div>
        <div class='result-details'>
            <span>Confidence: <b>{confidence:.2f}%</b></span>
            <span>Model Used: <b>{model_choice}</b></span>
        </div>
    </div>
    """
    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#111827') # Dark background for the figure
    ax.pie(
        probabilities, 
        labels=['Fake News', 'Real News'], 
        autopct='%1.1f%%',
        colors=['#F87171', '#34D399'],
        textprops={'color': 'white', 'fontweight': 'bold', 'fontsize': 12},
        wedgeprops={'edgecolor': '#1F2937', 'linewidth': 2},
        startangle=90
    )
    plt.title('Prediction Probability', color='white', fontsize=14, weight='bold')
    
    return output_message, fig

# Custom CSS for the modern look
custom_css = """
/* --- General & Theme --- */
body, .gradio-container { background-color: #030712 !important; color: #F9FAFB; }
.gradio-container { border: none !important; }
footer { display: none !important; }

/* --- Sidebar Navigation --- */
#sidebar { background-color: #111827; border-right: 1px solid #374151; padding: 1.5rem 0.75rem; min-height: 100vh; }
#nav-header { font-size: 1.8em; font-weight: bold; color: #E5E7EB; text-align: center; margin-bottom: 1.25rem; letter-spacing: 1px; }
.nav-btn { background: linear-gradient(45deg, #374151, #1F2937); color: #D1D5DB; border: 1px solid #4B5563; border-radius: 12px; margin: 0.5rem 0; font-weight: 600; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
.nav-btn:hover { background: linear-gradient(45deg, #4f46e5, #7c3aed); color: white; border-color: #6d28d9; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(79, 70, 229, 0.4); }

/* --- Main Content Area & Tabs --- */
#main-content { padding: 0 !important; }
#tabs-container > .tab-nav { display: none !important; } /* Hide the default tab buttons */
.gap { padding: 1.5rem !important; } /* Reduced padding */

/* --- Home Page Hero --- */
#home-hero-container { position: relative; height: 100%; background-color: #1F2937; width: 100%; display: flex; align-items: center; justify-content: center; border-radius: 18px; box-shadow: 0 8px 25px rgba(0,0,0,0.5); overflow: hidden; background-size: cover; background-position: center; min-height: 500px; }
.hero-overlay { background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(8px); padding: 2rem 2.5rem; border-radius: 16px; text-align: center; z-index: 2; box-shadow: 0 6px 20px rgba(0,0,0,0.3); }
.hero-title { font-size: 3.2em; color: #FFFFFF; text-shadow: 2px 2px 8px rgba(0,0,0,0.7); margin-bottom: 0.5em; font-weight: 800; }
.hero-subtitle { font-size: 1.5em; color: #D1D5DB; font-style: italic; opacity: 0.95; margin-top: 0; }

/* --- Features Page --- */
#features-title { color: #E5E7EB; font-size: 2em; font-weight: 700; border-bottom: 2px solid #4f46e5; padding-bottom: 0.5rem; margin-bottom: 1.25rem; }
#input-box textarea { background-color: #1F2937 !important; color: #E5E7EB !important; border: 1px solid #4B5563; border-radius: 12px; font-size: 1rem; }
#predict-btn { background: linear-gradient(45deg, #4f46e5, #7c3aed); color: white; font-weight: bold; border-radius: 10px; transition: all 0.3s ease; }
#predict-btn:hover { transform: scale(1.05); box-shadow: 0 4px 15px rgba(79, 70, 229, 0.5); }
.example-btn { background-color: #374151 !important; color: #D1D5DB !important; border-color: #4B5563 !important; }
.result-container { background-color: #1F2937; border: 1px solid #374151; border-radius: 12px; padding: 1.25rem; text-align: center; }
.result-label { font-size: 1.8rem; font-weight: 800; margin-bottom: 1rem; }
.result-details { display: flex; justify-content: space-around; font-size: 1rem; color: #D1D5DB; }
#plot-output { background-color: #111827 !important; border-radius: 12px !important; border: 1px solid #374151 !important; }

/* --- About Page --- */
.about-section { background-color: #1F2937; padding: 20px; border-radius: 14px; margin-bottom: 15px; border: 1px solid #374151; }
.about-section h3 { color: #c7d2fe; font-size: 1.4em; margin-top: 0; border-bottom: 1px solid #4f46e5; padding-bottom: 8px; }
.about-section li { color: #D1D5DB; margin-bottom: 8px; font-size: 1rem; }
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css, title="Fake News Detector") as dashboard:
    
    # REMOVED: image_url_state = gr.State(f"/file={IMAGE_FILENAME}")

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar"):
            gr.Markdown("## News Verifier", elem_id="nav-header")
            btn_home = gr.Button("🏠 Home", elem_classes="nav-btn")
            btn_features = gr.Button("🔎 Detector", elem_classes="nav-btn")
            btn_about = gr.Button("ℹ️ About", elem_classes="nav-btn")

        with gr.Column(scale=4, elem_id="main-content"):
            with gr.Tabs(elem_id="tabs-container") as tabs:
                # --- Home Page Tab ---
                with gr.Tab("Home", id="home", elem_id="home-tab"):
                    with gr.Column(elem_classes="gap"):
                        # Simple hero section without background image
                        gr.Markdown("""
                        <div class="hero-overlay">
                            <h1 class="hero-title">Fake News Detection and Prevention.</h1>
                            <p class="hero-subtitle">Empowering leaders with truth and technology.</p>
                        </div>
                        """)

                # --- Features Page Tab ---
                with gr.Tab("Detector", id="detector", elem_id="features-tab"):
                    with gr.Column(elem_classes="gap"):
                        gr.Markdown("### 🧠 Analyze News Authenticity", elem_id="features-title")
                        with gr.Row():
                            with gr.Column(scale=2):
                                input_text = gr.Textbox(label="Paste News Article Here", placeholder="The story begins...", lines=10, elem_id="input-box")
                                model_radio = gr.Radio(choices=["Logistic Regression", "Naive Bayes"], label="Select Analysis Model", value="Logistic Regression")
                                predict_btn = gr.Button("Verify Authenticity", elem_id="predict-btn")
                                gr.Markdown("#### Or try an example:")
                                with gr.Row():
                                    ex1 = gr.Button("Example 1 (Real)", elem_classes="example-btn")
                                    ex2 = gr.Button("Example 2 (Fake)", elem_classes="example-btn")
                            with gr.Column(scale=1):
                                output_result = gr.Markdown(label="Analysis Result")
                                chart = gr.Plot(label="Probability Distribution", elem_id="plot-output")

                # --- About Page Tab ---
                with gr.Tab("About", id="about", elem_id="about-tab"):
                    with gr.Column(elem_classes="gap"):
                        gr.Markdown("""
                        # ℹ️ About This Application

                        <div class="about-section">
                            <h3>🛠️ How It Works</h3>
                            <ol>
                                <li>Navigate to the <b>Detector</b> tab using the sidebar.</li>
                                <li>Paste the full text of a news article into the text box.</li>
                                <li>Choose between two machine learning models for the analysis.</li>
                                <li>Click the 'Verify Authenticity' button to process the text.</li>
                                <li>The system will classify the article as Real or Fake and show a confidence score.</li>
                            </ol>
                        </div>

                        <div class="about-section">
                            <h3>👩‍💻 Project Team</h3>
                            <ul>
                              <li>Kondaveeti Satwika (22bq1a4278@vvit.net)</li>
                              <li>Jakka Murali Karthik (22bq1a4263@vvit.net)</li>
                              <li>Munagala Geeth Sahith (22bq1a4297@vvit.net)</li>
                              <li>Narne Sri Sowmya (22bq1a42a0@vvit.net)</li>
                            </ul>
                        </div>
                                    
                        """)
    

    # Sidebar buttons control which tab is selected by its ID
    btn_home.click(lambda: gr.update(selected="home"), outputs=tabs)
    btn_features.click(lambda: gr.update(selected="detector"), outputs=tabs)
    btn_about.click(lambda: gr.update(selected="about"), outputs=tabs)

    # Prediction button event
    predict_btn.click(predict_news, inputs=[input_text, model_radio], outputs=[output_result, chart])

    # Example button events
    ex1_text = "WASHINGTON (Reuters) - The U.S. Supreme Court on Monday declined to hear a challenge to a lower court ruling that upheld the Affordable Care Act's requirement for most Americans to obtain health insurance or pay a penalty."
    ex2_text = "BREAKING: A new study from the Institute of Fictional Science claims that eating pizza three times a day is the secret to a long and healthy life, a fact they say has been suppressed by the broccoli industry for years."
    ex1.click(lambda: gr.update(value=ex1_text), outputs=input_text)
    ex2.click(lambda: gr.update(value=ex2_text), outputs=input_text)

# --- Run the Application ---
if __name__ == "__main__":
    # REMOVED: image existence check
    dashboard.launch(server_name="127.0.0.1", server_port=7861, allowed_paths=["."])
