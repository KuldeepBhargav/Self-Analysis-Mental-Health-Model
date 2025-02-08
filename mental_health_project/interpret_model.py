import joblib
import shap
import pandas as pd

model = joblib.load("mental_health_model.pkl")
df = pd.read_csv("processed_survey.csv")
X = df.drop(columns=['mental_health_condition'])
explainer = shap.Explainer(model)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)