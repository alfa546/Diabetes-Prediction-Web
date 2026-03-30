#  AI-Powered Diabetes Detection

A machine learning web application for diabetes risk prediction, with bilingual support (English & Urdu) and optional AI-powered health advice via Claude.
Built with: Python · Flask · Scikit-learn (Random Forest) ·

Project Structure
diabetes-project/
├── app.py                  ← Flask application (entry point)
├── requirements.txt        ← Python dependencies
├── model/
│   ├── train_model.py      ← ML model training script
│   ├── diabetes_model.pkl  ← Trained model (auto-generated)
│   └── scaler.pkl          ← Feature scaler (auto-generated)
└── templates/
    └── index.html          ← Frontend interface

Getting Started
Prerequisites

Python 3.10 or higher — download from python.org
During installation on Windows, make sure to check "Add Python to PATH"


Step 1 — Install Dependencies
Open a terminal in the project root directory and run:
bashpip install -r requirements.txt
Step 2 — Train the ML Model
bashcd model
python train_model.py
cd ..
This will generate diabetes_model.pkl and scaler.pkl in the model/ directory.
Step 3 — Start the Server
bashpython app.py
Step 4 — Open in Browser
Navigate to: http://localhost:5000

AI-Powered Advice (Optional)
To enable personalized AI recommendations:

Visit anthropic.com and create an account
Go to API Keys and generate a new key
Copy the key and paste it into the API Key field on the website
The app will now provide Claude-powered health insights alongside predictions


Model Details
PropertyValueAlgorithmRandom Forest ClassifierAccuracy94.5%DatasetPIMA Indians Diabetes (style)Input FeaturesGlucose, BMI, Age, Blood Pressure, Insulin, Skin Thickness, Pregnancies, Diabetes Pedigree Function
