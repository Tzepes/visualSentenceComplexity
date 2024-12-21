import pandas as pd
import matplotlib.pyplot as plt

# Load tfidf data from the CSV file
tfidf_df = pd.read_csv("tfidfData.csv")

# Plot all sentences
plt.figure(figsize=(15, 8))
for i in range(len(tfidf_df)):
    sentence_scores = tfidf_df.iloc[i]
    sorted_scores = sentence_scores[sentence_scores > 0].sort_values(ascending=False).head(10)  # Top 10 non-zero features
    plt.plot(sorted_scores.index, sorted_scores.values, marker="o", linestyle="--", label=f"Sentence {i + 1}")

plt.title("TF-IDF Scores for Top Features Across Sentences")
plt.xlabel("Words")
plt.ylabel("TF-IDF Score")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
