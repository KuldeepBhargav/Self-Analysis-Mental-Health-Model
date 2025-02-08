# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import pandas as pd

# df = pd.read_csv("processed_survey.csv")
# X = df.drop(columns=['mental_health_condition'])
# y = df['mental_health_condition']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# joblib.dump(model, "mental_health_model.pkl")

# print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
# print(classification_report(y_test, model.predict(X_test)))

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import pandas as pd

# # Load preprocessed data
# df = pd.read_csv("processed_survey.csv")

# # Updated feature list to include more relevant inputs
# X = df[['age', 'Gender_Male', 'work_interfere', 'family_history', 'self_employed', 'treatment']]
# y = df['mental_health_condition']  # Target variable

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train RandomForest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save model and feature names
# joblib.dump((model, X.columns.tolist()), "mental_health_model.pkl")

# # Print model performance
# print("Model Training Complete ✅")
# print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
# print(classification_report(y_test, model.predict(X_test)))

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# # Load dataset
# df = pd.read_csv("processed_survey.csv")

# # Print dataset columns for debugging
# print("Available Columns:", df.columns.tolist())

# # Standardize column names
# df.columns = df.columns.str.lower().str.replace(" ", "_")

# # Create missing columns
# if 'gender' in df.columns:  
#     df['gender_male'] = df['gender'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
# else:
#     print("⚠️ Skipping gender processing as 'gender' column is missing.")

# df['work_interfere'] = df['work_interfere'].map({
#     "No": 0, "Sometimes": 1, "Often": 2, "Always": 3
# })

# df['family_history'] = df['family_history'].map({"No": 0, "Yes": 1})
# df['self_employed'] = df['self_employed'].map({"No": 0, "Yes": 1})
# df['treatment'] = df['treatment'].map({"No": 0, "Yes": 1})

# # Remove NaN values
# df = df.dropna()

# # Define features and target
# if df.empty:
#     raise ValueError("🚨 ERROR: DataFrame is empty after preprocessing! Fix preprocessing.")
# if "gender" not in df.columns:
#     raise KeyError("🚨 ERROR: 'gender' column is missing! Fix dataset or preprocessing.")

# X = df[['age', 'self_employed', 'family_history', 'work_interfere', 'mental_health_condition', 'gender_male']]
# y = df['treatment']

# print("X shape:", X.shape)  # Ensure it's not (0, X)
# print("y shape:", y.shape)  # Ensure it's not (0,)

# if X.shape[0] == 0 or y.shape[0] == 0:
#     raise ValueError("🚨 ERROR: X or y is empty! Check preprocessing.")

# if X.shape[0] == 0 or y.shape[0] == 0:
#     raise ValueError("🚨 ERROR: X or y is empty! Check your data preprocessing.")

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save model and feature names
# joblib.dump((model, X.columns.tolist()), "mental_health_model.pkl")

# print("✅ Model training complete! Model saved as 'mental_health_model.pkl'")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("processed_survey.csv")  # Replace with your dataset file

# Separate features and target variable
X = df.drop(columns=['age', 'self_employed', 'family_history', 'work_interfere', 'mental_health_condition', 'gender_male'])  # Replace "target_column" with actual target column name
y = df["treatment"]  # Replace "target_column" with actual target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))  # Encode training data
    X_test[col] = le.transform(X_test[col].astype(str))  # Encode test data
    label_encoders[col] = le  # Save encoder for future use

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump((model, X_train.columns.tolist()), "mental_health_model.pkl")

joblib.dump(label_encoders, "label_encoders.pkl")  # Save encoders for future use

print("Model trained and saved successfully!")
