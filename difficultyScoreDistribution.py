import matplotlib.pyplot as plt
import pandas as pd

train_scores = pd.read_csv("./data/train.csv")

difficulty_scores = train_scores.iloc[:, -1]

plt.figure(figsize=(16, 10))
plt.hist(difficulty_scores, bins=20, color='blue', edgecolor='black')
plt.title("Histogram Distribution of Target Values")
plt.xlabel("Target")
plt.ylabel("Total Number of Observations")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()