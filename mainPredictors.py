import csv
import numpy as np
import pandas as pd
import nltk
import re
import textstat
import seaborn as sns
from scipy.stats import spearmanr

from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, Tree
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from corpus_data.colors_corpus import color_words
from corpus_data.wordsOfNums_corpus import words_of_numbers
from corpus_data.prepositions_corpus import prepositions
from corpus_data.positioning_corpus import position_words

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
val_df = pd.read_csv("./data/val.csv")


# nltk.download('punkt_tab')
# nltk.download("stopwords")
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('brown')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def sentence_preprocessing(sentence):
    sentence = sentence.lower()
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in stopwords.words("english")]
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

train_ids = train_df.iloc[:, 0].tolist()
train_sentences = train_df.iloc[:, 1].tolist()
train_scores = train_df.iloc[:, -1].tolist()

test_ids = test_df.iloc[:, 0].tolist()
test_sentences = test_df.iloc[:, 1].tolist()

valid_ids = val_df.iloc[:, 0].tolist()
valid_sentences = val_df.iloc[:, 1].tolist()
valid_scores = val_df.iloc[:, -1].tolist()

train_sentences_clean = [sentence_preprocessing(sentence) for sentence in train_sentences]
test_sentences_clean = [sentence_preprocessing(sentence) for sentence in test_sentences]
validation_sentences_clean = [sentence_preprocessing(sentence) for sentence in valid_sentences]

train_ids_df = pd.DataFrame(train_ids, columns=['train_ids'])
train_sentences_df = pd.DataFrame(train_sentences_clean, columns=['train_sentences'])
train_scores_df = pd.DataFrame(train_scores, columns=['train_scores'])

valid_ids_df = pd.DataFrame(valid_ids, columns=['valid_ids'])
valid_sentences_df = pd.DataFrame(validation_sentences_clean, columns=['valid_sentences'])

brown_words = brown.words()
brown_freq_dist = nltk.FreqDist(brown_words)
total_brown_words = len(brown_words)
stop_words = set(stopwords.words('english'))

def ExtractFeatures(sentences):
    features = {
        'word_rarity': WordRarity(sentences),
        'adjective_count': AdjectiveCount(sentences),
        'adverb_count': AdverbCount(sentences)
    }
    return features

def CharacterCount(sentences):
    character_count = []
    for sentence in sentences:
        removed_punctuation = re.sub(r'[^\w\s]', '', sentence)
        removed_spaces = re.sub(r'\s', '', removed_punctuation)
        character_count.append(len(removed_spaces))
    return np.array(character_count).reshape(-1, 1)

def WordRarity(sentences):
    word_rarity = []
    for sentence in sentences:
        sentence_rarity = []
        for word in word_tokenize(sentence):
            if word.isalpha() and word.lower() not in stop_words:
                sentence_rarity.append(-np.log((brown_freq_dist[word.lower()] + 1) / total_brown_words))
        word_rarity.append(np.mean(sentence_rarity) if sentence_rarity else 0)
    return np.array(word_rarity).reshape(-1, 1)

def AdjectiveCount(sentences):
    adjective_count = []  
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if any(synset.pos() == 'a' for synset in wn.synsets(word)):
                count += 1
        adjective_count.append(count)
    return np.array(adjective_count).reshape(-1, 1)

def NounCount(sentences):
    noun_count = []
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if any(synset.pos() == 'n' for synset in wn.synsets(word)):
                count += 1
        noun_count.append(count)
    return np.array(noun_count).reshape(-1, 1)

def ColorCount(sentences):
    colors_count = []
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if word in color_words:
                count += 1
        colors_count.append(count)
    return np.array(colors_count).reshape(-1, 1)

def WordLengthRatio(sentences):
    word_length_ratio = []
    for sentence in sentences:
        word_length_ratio.append(textstat.avg_letter_per_word(sentence))
    return np.array(word_length_ratio).reshape(-1, 1)

def PrepositionsCount(sentences):
    prepositions_count = []
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if word in prepositions:
                count += 1
        prepositions_count.append(count)
    return np.array(prepositions_count).reshape(-1, 1)

def DescriptivePositioning(sentences):
    position_count = []
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if word in position_words:
                count += 1
        position_count.append(count)
    return np.array(position_count).reshape(-1, 1)

def ParseTreeDepth(sentences):
    tree_depth = []
    for sentence in sentences:
        tree = Tree.fromstring(sentence)
        tree_depth.append(tree.height())
    return np.array(tree_depth).reshape(-1, 1)

def VerbCount(sentences):
    verb_count = []
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if any(synset.pos() == 'v' for synset in wn.synsets(word)):
                count += 1
        verb_count.append(count)
    return np.array(verb_count).reshape(-1, 1)

def AdverbCount(sentences):
    adverb_count = []
    for sentence in sentences:
        count = 0
        for word in word_tokenize(sentence):
            if any(synset.pos() == 'r' for synset in wn.synsets(word)):
                count += 1
        adverb_count.append(count)
    return np.array(adverb_count).reshape(-1, 1)

train_features = ExtractFeatures(train_sentences_clean)
test_features = ExtractFeatures(test_sentences_clean)
valid_features = ExtractFeatures(validation_sentences_clean)

X_train_features = np.hstack([train_features[feature] for feature in train_features])
X_test_features = np.hstack([test_features[feature] for feature in test_features])
X_valid_features = np.hstack([valid_features[feature] for feature in valid_features])

train_features_df = pd.DataFrame(X_train_features, columns=train_features.keys())
test_features_df = pd.DataFrame(X_test_features, columns=test_features.keys())
valid_features_df = pd.DataFrame(X_valid_features, columns=valid_features.keys())

plotting_of_featuresDF = pd.concat([train_features_df, train_scores_df], axis = 1)
axis = sns.pairplot(data = plotting_of_featuresDF, plot_kws = dict(color = "maroon"))
plt.show()

train_features_df['train_scores'] = train_scores
featureCorrelation = train_features_df.corr()
sns.heatmap(featureCorrelation, annot = True, cmap = 'Blues', linewidths = 1,
           annot_kws = {"weight": "bold", "fontsize": 10})
plt.figure(figsize = (10, 10))
plt.show()

train_features_df['sentence'] = train_sentences_clean
train_features_df['sentence_id'] = train_ids

# test_features_df['sentence'] = test_sentences_clean
# test_features_df['sentence_id'] = test_ids

# valid_features_df['sentence'] = validation_sentences_clean
# valid_features_df['sentence_id'] = valid_ids

print("Training Features Table:")
print(train_features_df.head())

# print("\nValidation Features Table:")
# print(valid_features_df.head())

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")

X_train_tfidf = vectorizer.fit_transform(train_sentences)
X_test_tfidf = vectorizer.transform(test_sentences)
X_valid_tfidf = vectorizer.transform(valid_sentences)

# Convert sparse matrix to DataFrame for visualization
# tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
# Save tfidf_df to a CSV file
# tfidf_df.to_csv("tfidfData.csv", index=False)

X_train_combined = np.hstack([X_train_features, X_train_tfidf.toarray()])
X_test_combined = np.hstack([X_test_features, X_test_tfidf.toarray()])
X_valid_combined = np.hstack([X_valid_features, X_valid_tfidf.toarray()])

X_train, X_test, y_train, y_test = train_test_split(X_train_combined, train_scores, test_size=0.2, random_state=42)

GBR_pipeline = Pipeline(steps = [("scaler", StandardScaler(with_mean=True)), ("regressor", GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42, verbose=1))])
GBR_pipeline.fit(X_train, y_train)

train_score = GBR_pipeline.score(X_train, y_train)
test_score = GBR_pipeline.score(X_test, y_test)

print(f"Train score: {train_score}")
print(f"Test score: {test_score}")

valid_predictions_GBR = GBR_pipeline.predict(X_valid_combined)
spearman_corr, p_value = spearmanr(valid_scores, valid_predictions_GBR)
print(f"Spearman Correlation: {spearman_corr}")
print(f"P-value: {p_value}")

test_predictions_GBR = GBR_pipeline.predict(X_test_combined)

GBR_predictions_df = pd.DataFrame({
    "id": test_ids,
    "score": test_predictions_GBR
})
GBR_predictions_df.to_csv("GBR_predictions.csv", index=False)

bayesianRidge = linear_model.BayesianRidge(verbose=1)
bayesianRidge.fit(X_train, y_train)

valid_predictions_bayesian = bayesianRidge.predict(X_valid_combined)
spearman_corr, p_value = spearmanr(valid_scores, valid_predictions_bayesian)
print(f"Spearman Correlation: {spearman_corr}")
print(f"P-value: {p_value}")

test_predictions_bayesian = bayesianRidge.predict(X_test_combined)

Bayesian_predictions_df = pd.DataFrame({
    "id": test_ids,
    "score": test_predictions_bayesian
})
Bayesian_predictions_df.to_csv("bayesianRidge_predictions.csv", index=False)