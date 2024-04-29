import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the tokenizer function
def makeTokens(f):
    tokens_by_slash = str(f.encode('utf-8')).split('/')
    total_tokens = []
    for i in tokens_by_slash:
        tokens = str(i).split('-')
        tokens_by_dot = []
        for j in tokens:
            temp_tokens = str(j).split('.')
            tokens_by_dot += temp_tokens
        total_tokens += tokens + tokens_by_dot
    total_tokens = list(set(total_tokens))
    if 'com' in total_tokens:
        total_tokens.remove('com')
    return total_tokens

# Load the logistic regression model and vectorizer
logit_model = joblib.load('url_classifier_model.pkl')
logit_vectorizer = joblib.load('vectorizer.pkl')

# Load the LightGBM model and vectorizer
lgbm_model = joblib.load('lgbm_url_classifier_model.pkl')
lgbm_vectorizer = joblib.load('lgbm_vectorizer.pkl')

# Load the TensorFlow model saved as .h5 and its vectorizer
tf_model = tf.keras.models.load_model('tf_url_classifier_model.h5')  # Make sure the path is correct
tf_vectorizer = joblib.load('tf_vectorizer.pkl')

# Function to get predictions
def get_predictions(url, vectorizer, model, is_tensorflow=False):
    # Transform the URL using the trained vectorizer
    X = vectorizer.transform([url])
    # Check if the model is a TensorFlow model
    if is_tensorflow:
        # Convert to dense array for TensorFlow if necessary
        X = X.toarray()
        prediction = model.predict(X)
        # Convert TensorFlow output to label
        prediction = 'bad' if prediction[0][0] > 0.5 else 'not malicious'
    else:
        # Use scikit-learn models
        prediction = model.predict(X)
        # Convert prediction to label
        prediction = 'bad' if prediction[0] == 'bad' else 'not malicious'
    return prediction

# URLs to test
urls_to_predict = [
    "https://chat.openai.com/c/d834d6e5-61f2-41bf-a93e-d59df1a28dab",
    "https://lms.manhattan.edu/my/",
    "https://www.google.com/search?q=ha&sca_esv=f3a10901b51afbdb&sca_upv=1&source=hp&ei=RIUsZp-8JOG05NoPm_OZ4AY&iflsig=ANes7DEAAAAAZiyTVGoc204O0Lb8073Kspdmr_grRs77&ved=0ahUKEwif7oekzeGFAxVhGlkFHZt5BmwQ4dUDCBc&uact=5&oq=ha&gs_lp=Egdnd3Mtd2l6IgJoYTILEC4YgAQYsQMYgwEyERAuGIAEGLEDGNEDGIMBGMcBMggQLhiABBixAzIIEC4YgAQYsQMyCBAAGIAEGLEDMgUQABiABDIOEC4YgAQYsQMYxwEYrwEyCxAAGIAEGLEDGIMBMgUQABiABDIFEAAYgARI2gJQAFiQAXAAeACQAQCYAYABoAHXAaoBAzEuMbgBA8gBAPgBAZgCAqAC6gHCAg4QLhiABBixAxiDARiKBZgDAJIHAzEuMaAHxR4&sclient=gws-wiz",
    "male.com/mal.exe"
]

# Display predictions for each URL and each model
for url in urls_to_predict:
    logit_pred = get_predictions(url, logit_vectorizer, logit_model)
    lgbm_pred = get_predictions(url, lgbm_vectorizer, lgbm_model)
    tf_pred = get_predictions(url, tf_vectorizer, tf_model, is_tensorflow=True)
    print(f"URL: {url}\n  Logistic Regression Prediction: {logit_pred}\n  LightGBM Prediction: {lgbm_pred}\n  TensorFlow Prediction: {tf_pred}\n")

