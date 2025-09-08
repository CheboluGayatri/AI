# sentiment analysis on text data using NLTK

import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

# Download necessary NLTK data
nltk.download("movie_reviews")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Prepare stopwords
stop_words = set(stopwords.words("english"))

# Prepare documents as (words, category) tuples
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle documents for random distribution
random.shuffle(documents)

# Feature extractor: convert words to dictionary of boolean features
def extract_features(words):
    return {word.lower(): True for word in words if word.isalpha() and word.lower() not in stop_words}

# Create feature sets
featuresets = [(extract_features(words), category) for (words, category) in documents]

# Train/test split
train_set, test_set = featuresets[:1600], featuresets[1600:]

# Train Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate classifier accuracy
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show most informative features
classifier.show_most_informative_features(10)

# Function to analyze sentiment of new text
def analyze_sentiment(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    features = extract_features(words)
    return classifier.classify(features)

# Test classifier on new sentences
test_sentences = [
    "I love this movie!",
    "This film was terrible.",
    "What a great experience!",
    "I didn't like the plot."
]

for sentence in test_sentences:
    print(f"Sentence: {sentence}")
    print(f"Predicted sentiment: {analyze_sentiment(sentence)}\n")
