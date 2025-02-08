import pandas as pd
import numpy as np

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Handle Gender Column
    if 'gender' in df.columns:
        df['gender_male'] = df['gender'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
    else:
        print("⚠️ Column 'gender' not found! Check your dataset.")

    # Mapping categorical values to numerical
    mappings = {
        'work_interfere': {"No": 0, "Sometimes": 1, "Often": 2, "Always": 3},
        'family_history': {"No": 0, "Yes": 1},
        'self_employed': {"No": 0, "Yes": 1},
        'treatment': {"No": 0, "Yes": 1}
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1)  # Replace NaN with -1
        else:
            print(f"⚠️ Column '{col}' not found! Skipping this step.")

    if 'mental_health_condition' not in df.columns:
        raise ValueError("Column 'mental_health_condition' is missing from dataset. Check survey.csv")

    # Convert remaining categorical columns to numerical
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'mental_health_condition']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df.to_csv("processed_survey.csv", index=False)  # Save cleaned dataset
    return df

if __name__ == "__main__":
    preprocess_data("survey.csv")
