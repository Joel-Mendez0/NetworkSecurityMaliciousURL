import joblib
from tensorflow.keras.models import load_model

def extract_features_from_url(url):
    # In practice, this should be replaced with the actual feature extraction logic.
    return [len(url), url.count('.')]

# Load the saved models
random_forest_model = joblib.load('/path/to/random_forest_model.joblib')
logistic_regression_model = joblib.load('/path/to/logistic_regression_model.joblib')
keras_model = load_model('/path/to/my_keras_model.h5')
urls_to_test = [
    'http://example1.com/malicious',
    'http://example2.com/benign'
]
for url in urls_to_test:
    features = extract_features_from_url(url)
    
    # Predict with RandomForest
    rf_prediction = random_forest_model.predict([features])
    
    # Predict with LogisticRegression
    lr_prediction = logistic_regression_model.predict([features])
    
    # Predict with Keras model
    keras_prediction = keras_model.predict([features])[0]  # Assuming binary classification with sigmoid

    # Interpret the Keras model prediction
    keras_pred_label = 1 if keras_prediction >= 0.5 else 0
    
    # Print the predictions
    print(f"URL: {url}")
    print(f"RandomForest Prediction: {rf_prediction[0]}")  # Assuming the prediction is a numpy array
    print(f"LogisticRegression Prediction: {lr_prediction[0]}")
    print(f"Keras Model Prediction: {keras_pred_label}")

