import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the data
urls_data = pd.read_csv("urldata.csv")

# Function to tokenize URLs
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
        total_tokens.remove('com')  # Remove 'com' as it's a common but uninformative token
    return total_tokens

# Preparing features and labels
y = urls_data['label']
url_list = urls_data['url']
vectorizer = TfidfVectorizer(tokenizer=makeTokens)
X = vectorizer.fit_transform(url_list)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM Classifier
lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
lgbm_classifier.fit(X_train, y_train)

# Evaluate the model
print(f"Training Accuracy: {lgbm_classifier.score(X_train, y_train):.2f}")
print(f"Testing Accuracy: {lgbm_classifier.score(X_test, y_test):.2f}")

# Save the model and vectorizer
joblib.dump(lgbm_classifier, 'lgbm_url_classifier_model.pkl')
joblib.dump(vectorizer, 'lgbm_vectorizer.pkl')

