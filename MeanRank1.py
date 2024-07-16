import pandas as pd
import numpy as np

# Read your CSV file containing AUC performances
# Make sure your CSV file has columns like 'Classifier', 'DataSet', 'Average'
# Rows should represent different combinations of classifiers and data sets

# Example CSV file structure:
# Classifier, DataSet, Average
# Classifier1, Dataset1, 0.85
# Classifier2, Dataset1, 0.82
# ...

# Read the CSV file
df = pd.read_csv('ranking_results.csv')

# Group by classifier and dataset, then rank based on AUC
# df['Rank'] = df.groupby('DataSet')['Average'].rank(method='min')
df['Rank'] = df.groupby('DataSet')['Average'].rank(method='min', ascending=False)
# Calculate mean rank for each classifier
mean_ranks = df.groupby('Classifier')['Rank'].mean().sort_values()

# Rank the classifiers based on their mean ranks
final_ranking = mean_ranks.rank(method='min').astype(int)

# Print mean ranks and final ranking
print("\nMean Ranks:")
print(mean_ranks)

print("\nFinal Ranking:")
print(final_ranking)
