import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('breast-cancer-wisconsin-data_data.csv')

# Define the 22 feature columns (replace these with your actual feature names)
feature_columns =["texture_mean","smoothness_mean","compactness_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","texture_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","texture_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",
]

# Ensure these feature columns are in the dataset
missing_features = [col for col in feature_columns if col not in data.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

# Define the target column
target_column = 'diagnosis'

# Prepare the feature matrix (X) and target vector (y)
X = data[feature_columns]
y = data[target_column].apply(lambda x: 1 if x == 'M' else 0)  # Assuming 'M' for malignant and 'B' for benign

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
model_filename = 'models/random_forest_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f'Model saved to {model_filename}')
