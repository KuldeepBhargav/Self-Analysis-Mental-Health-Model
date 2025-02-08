# import joblib
# import pandas as pd

# # Debugging: Check what is being loaded
# loaded_data = joblib.load("mental_health_model.pkl")
# print(type(loaded_data))  # Should be a tuple
# print(len(loaded_data))   # Should be 2

# def predict(symptoms):
#     model, feature_names = loaded_data
#     input_data = pd.DataFrame([symptoms], columns=feature_names)
#     return model.predict(input_data)[0]

# if __name__ == "__main__":
#     sample_input = {"age": 25, "Gender_Male": 1, "work_interfere": 0, "family_history": 1}
#     print("Predicted Condition:", predict(sample_input))


import joblib
import pandas as pd

def predict(symptoms):
    # Load and verify the model structure
    loaded_data = joblib.load("mental_health_model.pkl")

    # Debugging: Print structure of the loaded data
    print("Type of loaded_data:", type(loaded_data))
    print("Length of loaded_data:", len(loaded_data))
    print("Content Preview:", loaded_data)

    # Ensure correct unpacking
    if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
        model, feature_names = loaded_data
    else:
        raise ValueError("Unexpected data format in 'mental_health_model.pkl'. Expected (model, feature_names).")

    # Convert user input into DataFrame
    input_data = pd.DataFrame([symptoms], columns=feature_names)

    # Make prediction
    return model.predict(input_data)[0]

# Test prediction with a sample input
if __name__ == "__main__":
    sample_input = {
        "age": 30,
        "Gender_Male": 1,
        "work_interfere": 2,
        "family_history": 1,
        "self_employed": 0,
        "treatment": 1
    }
    print("Predicted Condition:", predict(sample_input))
