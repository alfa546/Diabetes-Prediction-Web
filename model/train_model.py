import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# =============================================
# PIMA Indians Diabetes Dataset (Embedded)
# =============================================
np.random.seed(42)

# Realistic PIMA-style data generation
def generate_pima_data(n=800):
    data = []
    for _ in range(n):
        diabetic = np.random.random() < 0.35
        if diabetic:
            pregnancies = np.random.randint(0, 15)
            glucose = np.random.normal(148, 25)
            blood_pressure = np.random.normal(74, 12)
            skin_thickness = np.random.normal(35, 12)
            insulin = np.random.normal(200, 100)
            bmi = np.random.normal(35, 7)
            dpf = np.random.normal(0.65, 0.3)
            age = np.random.normal(45, 10)
            label = 1
        else:
            pregnancies = np.random.randint(0, 8)
            glucose = np.random.normal(110, 20)
            blood_pressure = np.random.normal(68, 10)
            skin_thickness = np.random.normal(25, 10)
            insulin = np.random.normal(80, 50)
            bmi = np.random.normal(28, 5)
            dpf = np.random.normal(0.35, 0.2)
            age = np.random.normal(33, 8)
            label = 0

        data.append([
            max(0, int(pregnancies)),
            max(40, min(200, glucose)),
            max(30, min(130, blood_pressure)),
            max(0, min(60, skin_thickness)),
            max(0, min(600, insulin)),
            max(15, min(60, bmi)),
            max(0.05, min(2.5, dpf)),
            max(18, min(80, age)),
            label
        ])

    df = pd.DataFrame(data, columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ])
    return df

df = generate_pima_data(1000)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)

# Train Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Evaluate
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
gb_acc = accuracy_score(y_test, gb_model.predict(X_test_scaled))

print(f"Random Forest Accuracy: {rf_acc:.2%}")
print(f"Gradient Boosting Accuracy: {gb_acc:.2%}")

# Use best model
best_model = rf_model if rf_acc >= gb_acc else gb_model
best_acc = max(rf_acc, gb_acc)
print(f"\nBest Model Accuracy: {best_acc:.2%}")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

# Save model and scaler
model_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(best_model, os.path.join(model_dir, 'diabetes_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

print("\n✅ Model saved: diabetes_model.pkl")
print("✅ Scaler saved: scaler.pkl")
