import csv
import numpy as np
import pandas as pd
import nltk
import re
import textstat

import matplotlib
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
val_df = pd.read_csv("./data/val.csv")

nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()

def sentence_preprocessing(sentence):
    
    # Lower case text
    sentence = sentence.lower()
    # Remove special characters
    sentence = re.sub(r'[^a-z]', ' ', sentence)
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


# Extract the columns into lists
train_ids = train_df.iloc[:, 0].tolist()
train_sentences = train_df.iloc[:, 1].tolist()
train_scores = train_df.iloc[:, -1].tolist()  

test_ids = test_df.iloc[:, 0].tolist()
test_sentences = test_df.iloc[:, 1].tolist()

valid_ids = val_df.iloc[:, 0].tolist()
valid_sentences = val_df.iloc[:, 1].tolist()
valid_scores = val_df.iloc[:, -1].tolist()

# Clean the sentences
train_sentences_clean = [sentence_preprocessing(sentence) for sentence in train_sentences]
test_sentences_clean = [sentence_preprocessing(sentence) for sentence in test_sentences]

train_ids_df = pd.DataFrame(train_ids, columns=['train_ids'])
train_sentences_df = pd.DataFrame(train_sentences_clean, columns=['train_sentences'])
train_scores_df = pd.DataFrame(train_scores, columns=['train_scores'])

test_ids_df = pd.DataFrame(test_ids, columns=['test_ids'])
test_sentences_df = pd.DataFrame(test_sentences_clean, columns=['test_sentences'])


# Print the extracted data (optional)
print(train_ids_df.head())
print(train_sentences_df.head())
print(train_scores_df.head())
print(test_ids_df.head())
print(test_sentences_df.head())


# Features
def get_word_count(text):
    return np.array([len(sentence.split()) for sentence in text]).reshape(-1, 1)

def get_char_count(text):
    return np.array([len(sentence) for sentence in text]).reshape(-1, 1)

def get_avg_word_length(text):
    return np.array([np.mean([len(word) for word in sentence.split()]) for sentence in text]).reshape(-1, 1)

def get_lemma_count(text):
    return np.array([len(set(sentence.split())) for sentence in text]).reshape(-1, 1)

def get_syllable_count(text):
    return np.array([textstat.syllable_count(sentence) for sentence in text]).reshape(-1, 1)

# Combine features
X_train_features = np.hstack([
    get_word_count(train_sentences),
    get_char_count(train_sentences),
    get_avg_word_length(train_sentences),
    get_lemma_count(train_sentences),
    get_syllable_count(train_sentences)
])

X_test_features = np.hstack([
    get_word_count(test_sentences),
    get_char_count(test_sentences),
    get_avg_word_length(test_sentences),
    get_lemma_count(test_sentences),
    get_syllable_count(test_sentences)
])

# Create DataFrames for better readability
train_features_df = pd.DataFrame(X_train_features, columns=[
    'word_count', 'char_count', 'avg_word_length', 'lemma_count', 'syllable_count'
])

test_features_df = pd.DataFrame(X_test_features, columns=[
    'word_count', 'char_count', 'avg_word_length', 'lemma_count', 'syllable_count'
])

# Add the sentences for reference
train_features_df['sentence'] = train_sentences_clean
test_features_df['sentence'] = test_sentences_clean

# Display the table
print("Training Features Table:")
print(train_features_df.head())

print("\nTesting Features Table:")
print(test_features_df.head())

# TfidfVectorizer
vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )

# Apply TF-IDF transformation
X_train_tfidf = vectorizer.fit_transform(train_sentences)
X_test_tfidf = vectorizer.transform(test_sentences)

# Convert sparse matrix to DataFrame for visualization
# tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
# Save tfidf_df to a CSV file
# tfidf_df.to_csv("tfidfData.csv", index=False)

# Combine all features into one matrix
X_train_combined = np.hstack([X_train_features, X_train_tfidf.toarray()])
X_test_combined = np.hstack([X_test_features, X_test_tfidf.toarray()])

# Create Pipeline
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),  # Scale numerical features
        ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_train_combined, train_scores, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Evaluate the model
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

print(f"Train score: {train_score}")
print(f"Test score: {test_score}")

X_trainBayes, X_testBayes, y_trainBayes, y_testBayes = train_test_split(X_train_combined, train_scores, test_size=0.1, random_state=13)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()