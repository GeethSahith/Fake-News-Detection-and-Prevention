# Fake News Detector

This application uses machine learning to detect potentially fake news articles.

## Available Models

- **Naive Bayes**: Traditional probabilistic classifier using word frequencies
- **Logistic Regression**: Linear classifier with good interpretability and performance

## Running the Application

You have two options to run this application:

### Option 1: Using Gradio (Original Interface)

1. Double-click on `run_gradio_app.bat` or run the following command in your terminal:
   ```
   python app.py
   ```

2. Open your browser and navigate to: http://127.0.0.1:7860


## Troubleshooting

### Model Version Warnings

You may see warnings about model versions when running the application:

```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.0.2 when using version 1.6.0
```

These warnings indicate that the models were trained with an older version of scikit-learn than what you're currently using. While the application should still work, you may want to retrain the models with your current scikit-learn version for optimal performance.

### Application Not Opening Automatically

If the application starts but no browser window opens:
- For Gradio: Manually navigate to http://127.0.0.1:7860

### Other Issues

If you encounter other issues:
1. Make sure all required packages are installed:
   ```
   pip install -r requirements.txt
   ```
2. Check that all model files (.pkl) are present in the project directory
3. Try running the application with the debug flag:
   ```
   python app.py --debug
   ```

