---
title: Fake News Detection
emoji: 📰
colorFrom: yellow
colorTo: blue
sdk: streamlit
sdk_version: 1.35.0
app_file: streamlit_app.py
pinned: false
short_description: 📰 Fake News Classifier built with Streamlit
---

# Fake News Detector

This application uses machine learning to detect potentially fake news articles.

## Available Models

- **Naive Bayes**: Traditional probabilistic classifier using word frequencies  
- **Logistic Regression**: Linear classifier with good interpretability and performance

## Running the Application

1. Ensure all dependencies are installed:
    ```
    pip install -r requirements.txt
    ```
2. Make sure all model files (`.pkl`) are present in the project directory:
    - `lg_fake_news_model.pkl`
    - `nb_fake_news_model.pkl`
    - `tfidf_vectorizer.pkl`
3. Run the Streamlit app:
    ```
    streamlit run streamlit_app.py
    ```
## Creating and Using a Python Virtual Environment

To create a Python environment for your Streamlit fake news detector app:

1. **Open a terminal and create a virtual environment:**
    ```
    python -m venv .venv
    ```

2. **Activate the environment:**
    - On macOS/Linux:
      ```
      source .venv/bin/activate
      ```
    - On Windows:
      ```
      .venv\Scripts\activate
      ```

3. **Install all required dependencies:**
    ```
    pip install -r requirements.txt
    ```

Once the environment is set up and activated, you can run the app as described above.
