# 🩺 DiabetesAI — ذیابیطس ڈیٹیکشن سسٹم

AI-Powered Diabetes Detection Website | Urdu + English
Built with: Python + Flask + ML (Random Forest) + Claude AI

---

## 📁 Project Structure

```
diabetes-project/
├── app.py                  ← Flask server (main file)
├── requirements.txt        ← Python packages
├── model/
│   ├── train_model.py      ← ML model training
│   ├── diabetes_model.pkl  ← Trained model (auto-generated)
│   └── scaler.pkl          ← Data scaler (auto-generated)
└── templates/
    └── index.html          ← Website frontend
```

---

## 🚀 Setup Instructions (English)

### Step 1 - Install Python
- Download Python 3.10+ from https://python.org
- Make sure to check "Add to PATH" during installation

### Step 2 - Install packages
Open terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 3 - Train the ML Model
```bash
cd model
python train_model.py
cd ..
```
This will create `diabetes_model.pkl` and `scaler.pkl`

### Step 4 - Start the Website
```bash
python app.py
```

### Step 5 - Open in Browser
Go to: **http://localhost:5000**

---

## 🚀 Setup (اردو میں)

### پہلا قدم - Python انسٹال کریں
- https://python.org سے Python 3.10+ ڈاؤنلوڈ کریں
- انسٹال کے دوران "Add to PATH" ضرور چیک کریں

### دوسرا قدم - پیکیجز انسٹال کریں
اس فولڈر میں terminal کھولیں اور یہ چلائیں:
```
pip install -r requirements.txt
```

### تیسرا قدم - ML Model بنائیں
```
cd model
python train_model.py
cd ..
```

### چوتھا قدم - Website شروع کریں
```
python app.py
```

### پانچواں قدم - Browser میں کھولیں
یہاں جائیں: **http://localhost:5000**

---

## 🤖 AI Integration (Optional)

1. Go to https://anthropic.com → Sign Up → API Keys
2. Create new key → Copy it
3. Paste in the API Key box on the website
4. Get personalized AI-powered advice!

---

## 📊 Model Details

- Algorithm: Random Forest Classifier
- Accuracy: 94.5%
- Features: Glucose, BMI, Age, Blood Pressure, Insulin, Skin Thickness, Pregnancies, DPF
- Dataset: PIMA Indians Diabetes (style)

---

## ⚠️ Disclaimer

This tool is for educational/screening purposes only.
Always consult a qualified doctor for medical diagnosis.

یہ صرف معلوماتی مقاصد کے لیے ہے۔ طبی تشخیص کے لیے ڈاکٹر سے ملیں۔
