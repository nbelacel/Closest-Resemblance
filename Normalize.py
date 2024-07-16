import pandas as pd

# Function to normalize features between 0 and 1 using the specified formula
def normalize_features(data):
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data

# Function to read CSV file, normalize features (excluding the last column), and save the result
def normalize_csv(input_csv, output_csv):
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_csv)

    # Exclude the last column (class labels)
    features = df.iloc[:, :-1]

    # Normalize features using the specified formula
    normalized_features = features.apply(lambda x: normalize_features(x))

    # Combine the normalized features with the last column
    normalized_data = pd.concat([normalized_features, df.iloc[:, -1]], axis=1)

    # Save the normalized data to a new CSV file
    normalized_data.to_csv(output_csv, index=False)

# Example usage
input_csv_file = './datasets/Raisin.csv'  # Update with your actual file name
output_csv_file = './datasets/Raisin.csv'  # Update with your desired output file name

normalize_csv(input_csv_file, output_csv_file)
