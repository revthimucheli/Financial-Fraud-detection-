import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Define the dataset path
DATASET_PATH = 'creditcard.csv'

# Ensure the dataset exists
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

# Features and target
X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']]
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Train the model
dtree = DecisionTreeClassifier(criterion='entropy', random_state=42)
dtree.fit(X_train, y_train)

# Save the model and feature names together in a dictionary
model_data = {
    "model": dtree,
    "feature_names": X.columns.tolist()
}

# Save the dictionary to a joblib file
MODEL_PATH = 'financial_fraud_model.joblib'
joblib.dump(model_data, MODEL_PATH)
print(f"Model and feature names saved successfully as '{MODEL_PATH}'")
