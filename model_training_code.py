import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import re

# Function to clean text data
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    return text.lower()

# Load dataset with smaller chunks (to optimize memory usage)
chunksize = 100000  
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None, chunksize=chunksize)

# Define column names
col_names = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Initialize empty DataFrame to concatenate chunks
data_list = []
for chunk in df:
    chunk.columns = col_names
    data_list.append(chunk[['text', 'target']])

# Concatenate all chunks into a single DataFrame
df = pd.concat(data_list, ignore_index=True)

# Extract 'text' and 'target' columns
texts = df['text'].values
labels = df['target'].values

# Clean text data
texts = [clean_text(text) for text in texts]

# Convert target labels:
# 0 (negative) -> 0
# 2 (neutral)  -> 1
# 4 (positive) -> 2
labels = np.where(labels == 4, 2, labels)  # convert 4 to 2 (positive)
labels = np.where(labels == 2, 1, labels)  # convert 2 to 1 (neutral)
labels = np.where(labels == 0, 0, labels)  # keep 0 (negative)

# Hyperparameters
max_features = 10000  # Reduced vocabulary size for efficiency
sequence_length = 100  # Reduce sequence length
embedding_dim = 64  # Reduced embedding dimensions

# 1. Vectorization of Text Data using TextVectorization Layer
vectorizer = layers.TextVectorization(max_tokens=max_features, output_sequence_length=sequence_length)
vectorizer.adapt(texts)  # Build the vocabulary

# Vectorize the text data
X = vectorizer(np.array([[s] for s in texts])).numpy()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# 2. Define the Model Architecture
model = keras.Sequential([
    layers.Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length),  # Embedding Layer
    layers.Conv1D(64, 7, padding='valid', activation='relu', strides=2),  # Smaller Conv Layer for efficiency
    layers.GlobalMaxPooling1D(),  # Global max pooling
    layers.Dense(64, activation='relu'),  # Smaller Dense layer
    layers.Dropout(0.5),  # Dropout layer
    layers.Dense(3, activation='softmax')  # Output layer for 3 classes: positive, neutral, negative
])

# Compile the model with lower learning rate for better training
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Add EarlyStopping callback to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 3. Train the Model
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Saved the model to an HDF5 file
model.save('optimized_sentiment_model_7.h5')
print("Model saved as optimized_sentiment_model_7.h5")

# Loaded the model when needed
loaded_model = keras.models.load_model('optimized_sentiment_model_7.h5')
print("Model loaded successfully")

# 4. Evaluate the loaded model
score, acc = loaded_model.evaluate(X_test, y_test)
print(f"Test accuracy (loaded model): {acc}")

# Function to predict sentiment for a given statement
def predict_sentiment(statement):
    cleaned_statement = clean_text(statement)
    vectorized_statement = vectorizer([cleaned_statement])
    prediction = loaded_model.predict(vectorized_statement)
    
    sentiment_class = np.argmax(prediction, axis=1)[0]
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[sentiment_class]
