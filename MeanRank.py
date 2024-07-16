import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Read your CSV file containing AUC or accuracy performances
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

# Create a dictionary to store AUC or accuracy values for each classifier
metric_values = {classifier: [] for classifier in classifiers}

# Loop through classifiers and datasets to collect AUC or accuracy values
for classifier in classifiers:
    for dataset in datasets:
        # Extract AUC values for the current classifier and dataset
        metric_values[classifier].extend(df[(df['Classifier'] == classifier) & (df['DataSet'] == dataset)]['Average'].astype(float).values)

# Convert the metric values to NumPy arrays
metric_values_np = {classifier: np.array(metric_values[classifier]).astype(float) for classifier in classifiers}

# Normalize scores using Min-Max normalization
scaler = MinMaxScaler()
normalized_metric_values = {classifier: scaler.fit_transform(metric_values_np[classifier].reshape(-1, 1)).flatten() for classifier in classifiers}

# Calculate average normalized scores
average_normalized_scores = {classifier: np.mean(normalized_metric_values[classifier]) for classifier in classifiers}

# Rank algorithms based on average normalized scores
ranked_algorithms = sorted(average_normalized_scores.items(), key=lambda x: x[1], reverse=True)

# Print results
print("\nAverage Normalized Scores:")
for classifier, score in ranked_algorithms:
    print(f"{classifier}: {score}")

# Print Rankings
print("\nAlgorithm Rankings based on Average Normalized Scores:")
for rank, (classifier, _) in enumerate(ranked_algorithms, 1):
    print(f"Rank {rank}: {classifier}")

