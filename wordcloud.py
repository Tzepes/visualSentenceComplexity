import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the TF-IDF data
tfidf_df = pd.read_csv("tfidf_data.csv")

# Sum the TF-IDF scores across all sentences
word_scores = tfidf_df.sum(axis=0)

# Sort words by their total TF-IDF score
sorted_scores = word_scores.sort_values(ascending=False)

# Plot the top 20 words
top_words = sorted_scores.head(20)

# Seaborn plot
sns.barplot(x=top_words.values, y=top_words.index, palette="viridis")
plt.title("Top 20 TF-IDF Words")
plt.xlabel("TF-IDF Score")
plt.ylabel("Words")
plt.show()