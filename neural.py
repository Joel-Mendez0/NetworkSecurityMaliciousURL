import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib

# Load the data
urls_data = pd.read_csv("urldata.csv")

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
        total_tokens.remove('com')  # Remove 'com' as it's common but not informative
    return total_tokens

# Preparing features and labels
y = urls_data['label'].apply(lambda x: 1 if x == 'bad' else 0)
url_list = urls_data['url']
vectorizer = TfidfVectorizer(tokenizer=makeTokens, max_features=10000)
X = vectorizer.fit_transform(url_list)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

import tensorflow as tf

# Explicitly define a training step as a tf.function
@tf.function
def train_step(model, X_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(X_batch, training=True)
        loss = model.compiled_loss(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Custom training loop to handle sparse matrix
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    y_train = y_train.values
    y_test = y_test.values

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = tf.convert_to_tensor(X_train[i:i+batch_size].toarray())  # Convert sparse batch to dense tensor
            y_batch = tf.convert_to_tensor(y_train[i:i+batch_size])
            loss = train_step(model, X_batch, y_batch)
        print(f"Loss: {loss.numpy()}")

        # Evaluate the model on the test set
        scores = model.evaluate(X_test.toarray(), y_test, verbose=0)
        print(f"Testing Accuracy: {scores[1]*100:.2f}%")

# Ensure the model and training loop are correctly set up
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_model(model, X_train, y_train, X_test, y_test)

# Save the model and vectorizer
model.save('/home/joel/URLdetection/tf_url_classifier_model.h5')
joblib.dump(vectorizer, '/home/joel/URLdetection/tf_vectorizer.pkl')

