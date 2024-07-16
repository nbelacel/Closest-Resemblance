#import pandas as np
import numpy as np
from scipy.stats import wilcoxon

# Load your CSV file containing AUC performances
# Make sure your CSV file has columns like 'Classifier', 'DataSet', 'Fold1', 'Fold2', ..., 'Average'
# Rows should represent different combinations of classifiers and data sets

# Example CSV file structure:
# Classifier, DataSet, Fold1, Fold2, Fold3, Fold4, Fold5, Average
# Classifier1, Dataset1, 0.85, 0.88, 0.90, 0.87, 0.89, 0.878
# Classifier2, Dataset1, 0.82, 0.84, 0.88, 0.85, 0.87, 0.852
# ...

# Load data from CSV into a numpy array
data = np.genfromtxt('aucresults.csv', delimiter=',', dtype=None, names=True)
# Extract relevant columns
classifiers = np.unique(data['Classifier'])
datasets = np.unique(data['DataSet'])

# Create a dictionary to store the average AUCs for each classifier
average_auc = {classifier: [] for classifier in classifiers}

# Loop through classifiers and datasets to calculate average AUCs
for classifier in classifiers:
    for dataset in datasets:
        # Extract AUC values for the current classifier and dataset
        auc_values = data[(data['Classifier'] == classifier) & (data['DataSet'] == dataset)]['Average']
        # Calculate average AUC
        avg_auc = np.mean(auc_values)
        average_auc[classifier].append(avg_auc)

# Perform Wilcoxon signed-rank test and generate rankings
rankings = {}
for classifier1 in classifiers:
    for classifier2 in classifiers:
        if classifier1 != classifier2:
            # Perform Wilcoxon signed-rank test
            _, p_value = wilcoxon(average_auc[classifier1], average_auc[classifier2])
            # Determine the winner
            if p_value < 0.005:  # You can adjust the significance level as needed
                winner = classifier1 if np.mean(average_auc[classifier1]) > np.mean(average_auc[classifier2]) else classifier2
                loser = classifier2 if winner == classifier1 else classifier1
                rankings[winner] = rankings.get(winner, 0) + 1
                rankings[loser] = rankings.get(loser, 0)

# Print rankings
sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
print("Classifier Rankings:")
for classifier, score in sorted_rankings:
    print(f"{classifier}: {score} wins")


