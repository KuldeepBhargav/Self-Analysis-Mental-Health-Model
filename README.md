🧠 Self-Analysis Mental Health Model
🚀 A machine learning-based mental health condition predictor built with Streamlit and scikit-learn.

📌 Features
✅ Predicts potential mental health conditions based on user inputs.
✅ Uses a machine learning model trained on relevant health data.
✅ Interactive web app powered by Streamlit.
✅ Simple UI for easy user interaction.
✅ Supports real-time predictions.

🔧 Installation & Setup
Follow these steps to install dependencies and run the project:

1️⃣ Clone the Repository
git clone https://github.com/KuldeepBhargav/Self-Analysis-Mental-Health-Model.git
cd mental-health-model

2️⃣ Create a Virtual Environment
python -m venv venv

3️⃣ Activate the Virtual Environment
On Windows:
.\venv\Scripts\activate
On Mac/Linux:
source venv/bin/activate

4️⃣ Install Required Packages
pip install pandas numpy seaborn matplotlib scikit-learn joblib streamlit

🛠 Data Preprocessing & Model Training
5️⃣ Preprocess the Data
python data_preprocessing.py
This script cleans and prepares the dataset for training.

6️⃣ Train the Model
python train_model.py
This will train the model and save it as mental_health_model.pkl.

🚀 Running the Web App
7️⃣ Launch the Streamlit App
streamlit run app.py
This will start the Streamlit Web App in your browser.

🖥️ Expected Output
User enters details (age, gender, work interference, etc.).
Clicks "Predict 🏥" button.
The app displays a predicted mental health condition.

📜 Project Structure
mental-health-model/
│── data_preprocessing.py   # Data cleaning & preparation
│── train_model.py          # Training & saving the ML model
│── app.py                  # Streamlit web app
│── mental_health_model.pkl # Trained model
│── requirements.txt        # Dependencies
│── README.md               # Project documentation

📌 Notes
🔹 Ensure mental_health_model.pkl is generated before running app.py.
🔹 If the app fails, check if feature names match in train_model.py and app.py.
