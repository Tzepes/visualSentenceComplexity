import csv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

# Extract the columns into lists
train_ids = train_df.iloc[:, 0].tolist()
train_sentences = train_df.iloc[:, 1].tolist()
train_scores = train_df.iloc[:, -1].tolist()  

test_ids = test_df.iloc[:, 0].tolist()
test_sentences = test_df.iloc[:, 1].tolist()

# train_ids_df = pd.DataFrame(train_ids, columns=['train_ids'])
# train_sentences_df = pd.DataFrame(train_sentences, columns=['train_sentences'])
# train_scores_df = pd.DataFrame(train_scores, columns=['train_scores'])

# test_ids_df = pd.DataFrame(test_ids, columns=['test_ids'])
# test_sentences_df = pd.DataFrame(test_sentences, columns=['test_sentences'])


# # Print the extracted data (optional)
# print(train_ids_df.head())
# print(train_sentences_df.head())
# print(train_scores_df.head())
# print(test_ids_df.head())
# print(test_sentences_df.head())


# TfidfVectorizer
vectorizer = TfidfVectorizer()


# Features
def get_word_count(text):
    return np.array([len(sentence.split()) for sentence in text]).reshape(-1, 1)

def get_char_count(text):
    return np.array([len(sentence) for sentence in text]).reshape(-1, 1)

def get_avg_word_length(text):
    return np.array([np.mean([len(word) for word in sentence.split()]) for sentence in text]).reshape(-1, 1)

# Combine features
X_train_features = np.hstack([
    get_word_count(train_sentences),
    get_char_count(train_sentences),
    get_avg_word_length(train_sentences),
])

X_val_features = np.hstack([
    get_word_count(test_sentences),
    get_char_count(test_sentences),
    get_avg_word_length(test_sentences),
])

# Apply TF-IDF transformation
X_train_tfidf = vectorizer.fit_transform(train_sentences)
X_val_tfidf = vectorizer.transform(test_sentences)

# Combine all features into one matrix
X_train_combined = np.hstack([X_train_features, X_train_tfidf.toarray()])
X_val_combined = np.hstack([X_val_features, X_val_tfidf.toarray()])

# Create Pipeline
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),  # Scale numerical features
        ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ]
)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_train_combined, train_scores, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Evaluate the model
train_score = pipeline.score(X_train, y_train)
val_score = pipeline.score(X_val, y_val)

print(f"Train score: {train_score}")
print(f"Validation score: {val_score}")