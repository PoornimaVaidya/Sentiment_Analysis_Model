import numpy as np
import re
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Function to clean text data
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    return text.lower()

# Load the pre-trained sentiment model
loaded_model = keras.models.load_model('optimized_sentiment_model_7.h5')

# Recreate the TextVectorization layer with training parameters
max_features = 10000  # Should match the training hyperparameter
sequence_length = 100  # Should match the training hyperparameter
vectorizer = layers.TextVectorization(max_tokens=max_features, output_sequence_length=sequence_length)

# Load your training data to adapt the vectorizer
train_df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)
train_df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
train_texts = train_df['text'].values

# Clean and adapt the vectorizer
train_texts = [clean_text(text) for text in train_texts]
vectorizer.adapt(train_texts)

# Function to predict sentiment for a given statement
def predict_sentiment(statement):
    # Clean the input text
    cleaned_statement = clean_text(statement)
    
    # Vectorize the cleaned statement
    vectorized_statement = vectorizer([cleaned_statement])
    
    # Get the prediction from the loaded model
    prediction = loaded_model.predict(vectorized_statement)
    
    # Get the index of the class with the highest probability
    sentiment_class = np.argmax(prediction, axis=1)[0]
    
    # Map the class index to the corresponding sentiment
    sentiment_map = {0: "Negative", 2: "Neutral", 1: "Positive"}
    
    # Return the predicted sentiment
    return sentiment_map[sentiment_class]

# Load and preprocess the test dataset
# Uncomment and replace with your actual test data path
# test_df = pd.read_csv('path_to_your_test_data.csv', encoding='ISO-8859-1', header=None)
# test_df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
# X_test = test_df['text'].values
# y_test = test_df['target'].values

# Clean and vectorize test data
# X_test = [clean_text(text) for text in X_test]
# X_test = vectorizer(np.array([[s] for s in X_test])).numpy()

# Evaluate model performance
# y_pred = np.argmax(loaded_model.predict(X_test), axis=1)
# print("Classification report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example test statements to see individual predictions
test_statements = [
    "This app is super useful! It has all the features I need, and it's very easy to navigate",
    "These shoes are so comfortable and stylish! I wear them everywhere.",
    "This game is so much fun! The graphics are amazing, and it keeps me entertained for hours.",
    "Terrible experience. The product arrived damaged, and customer support was useless.",
    "This restaurant was a huge letdown. The food was cold, and the service was offensive.",
    "The instructions were unclear, and the assembly was a nightmare. Never buying from here again.",
    "The material of this shirt is cheap and uncomfortable. Not worth the price at all. ",

]

# Run predictions for each test case
for statement in test_statements:
    predicted_sentiment = predict_sentiment(statement)
    print(f"The sentiment for '{statement}' is: {predicted_sentiment}")
