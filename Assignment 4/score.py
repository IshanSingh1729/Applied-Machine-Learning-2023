import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib

# Load trained model
mlp_model = joblib.load("C:/Jupyter Lab/CMI/Applied Machine Learning/Assignment 3/model/multilayer-perceptron-model/model.pkl")

# Load data
train = pd.read_csv("C:/Jupyter Lab/CMI/Applied Machine Learning/Assignment 3/data/train.csv")
X_train = train["Message"]
X_train = X_train.replace(np.nan, '', regex=True)

# Preprocess the input text data and convert it into numeric features
count_vect = CountVectorizer().fit(X_train)
tfidf_transformer = TfidfTransformer().fit(count_vect.transform(X_train))

def score(text: str, model: MLPClassifier, threshold: float) -> (bool, float):
    """
    Scores a trained model on a text.

    Args:
        text (str): The text to be scored.
        model (MLPClassifier): The trained model to be used for scoring.
        threshold (float): The threshold above which the model will classify the text as positive.

    Returns:
        prediction (bool): The predicted class label of the text (True for positive, False for negative).
        propensity (float): The propensity (i.e., probability) of the text being classified as positive.
    """
    # Transform the input text using the already fitted transformers
    X_test_counts = count_vect.transform([text])
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # Get the predicted probabilities for the text from the model
    predicted_probabilities = model.predict_proba(X_test_tfidf)[0]

    # Calculate the propensity of the text being classified as positive
    propensity = predicted_probabilities[1]

    # Check if the propensity is above the given threshold
    if propensity >= threshold:
        # If the propensity is above the threshold, predict a positive label
        prediction = 1
    else:
        # If the propensity is below the threshold, predict a negative label
        prediction = 0

    return prediction, propensity
