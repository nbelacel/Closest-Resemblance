import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, rankdata, ranksums
from itertools import combinations

# Load your CSV file containing AUC performances
# Make sure your CSV file has columns like 'Classifier', 'DataSet', 'Fold1', 'Fold2', ..., 'Average'
# Rows should represent different combinations of classifiers and data sets

# Example CSV file structure:
# Classifier, DataSet, Fold1, Fold2, Fold3, Fold4, Fold5, Average
# Classifier1, Dataset1, 0.85, 0.88, 0.90, 0.87, 0.89, 0.878
# Classifier2, Dataset1, 0.82, 0.84, 0.88, 0.85, 0.87, 0.852
# ...

# Read the CSV file
df = pd.read_csv('aucresults.csv')
# Extract relevant columns
classifiers = df['Classifier'].unique()
datasets = df['DataSet'].unique()

# Create a dictionary to store AUC values for each classifier
auc_values = {classifier: [] for classifier in classifiers}

# Loop through classifiers and datasets to collect AUC values
for classifier in classifiers:
    for dataset in datasets:
        # Extract AUC values for the current classifier and dataset
        auc_values[classifier].extend(
            df[(df['Classifier'] == classifier) & (df['DataSet'] == dataset)]['Average'].values)

# Perform Friedman test
_, p_value_friedman = friedmanchisquare(*[auc_values[classifier] for classifier in classifiers])

if p_value_friedman < 0.02:
    print("Friedman test indicates significant differences among classifiers.")

    # Perform post-hoc analysis using Nemenyi test
    n_classifiers = len(classifiers)
    ranks = np.zeros((n_classifiers, len(auc_values[classifiers[0]])))

    for i, classifier in enumerate(classifiers):
        ranks[i] = rankdata(auc_values[classifier])

    # Calculate mean rank for each classifier
    mean_ranks = np.mean(ranks, axis=1)

    # Calculate critical difference for Nemenyi test
    q_val = 2.343  # You can adjust the critical value based on the number of classifiers and confidence level
    critical_difference = q_val * np.sqrt((n_classifiers * (n_classifiers + 1)) / (6 * len(auc_values[classifiers[0]])))

    # Compare each pair of classifiers using Nemenyi test
    rankings = {}
    for pair in combinations(classifiers, 2):
        _, p_value_nemenyi = ranksums(ranks[classifiers == pair[0]][0], ranks[classifiers == pair[1]][0])
        if p_value_nemenyi < critical_difference:
            winner = pair[0] if mean_ranks[classifiers == pair[0]][0] < mean_ranks[classifiers == pair[1]][0] else pair[
                1]
            loser = pair[1] if winner == pair[0] else pair[0]
            rankings[winner] = rankings.get(winner, 0) + 1
            rankings[loser] = rankings.get(loser, 0)

    # Print mean rank values and rankings
    print("\nMean Rank Values:")
    for classifier, mean_rank in zip(classifiers, mean_ranks):
        print(f"{classifier}: {mean_rank}")

    print("\nClassifier Rankings based on Mean Rank:")
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    for classifier, score in sorted_rankings:
        print(f"{classifier}: {score} wins")
else:
    print("Friedman test does not indicate significant differences among classifiers.")

