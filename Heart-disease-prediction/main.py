import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("C:/Users/ASUS/Downloads/Heart-Disease-Prediction-using-Machine-Learning-master/Heart-Disease-Prediction-using-Machine-Learning-master/heart.csv")

# Define features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for Flask usage
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Dictionary to store accuracies
model_accuracies = {}

# Logistic Regression
lr = LogisticRegression(class_weight="balanced", solver="liblinear")
lr.fit(X_train, y_train)
model_accuracies["Logistic Regression"] = accuracy_score(y_test, lr.predict(X_test)) * 100

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
model_accuracies["Naive Bayes"] = accuracy_score(y_test, nb.predict(X_test)) * 100

# Support Vector Machine
svm = SVC(kernel="linear", probability=True, class_weight="balanced")
svm.fit(X_train, y_train)
model_accuracies["SVM (Linear)"] = accuracy_score(y_test, svm.predict(X_test)) * 100

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
model_accuracies["KNN"] = accuracy_score(y_test, knn.predict(X_test)) * 100

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
model_accuracies["Decision Tree"] = accuracy_score(y_test, dt.predict(X_test)) * 100

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
model_accuracies["Random Forest"] = accuracy_score(y_test, rf.predict(X_test)) * 100

# XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, learning_rate=0.05)
xgb.fit(X_train, y_train)
model_accuracies["XGBoost"] = accuracy_score(y_test, xgb.predict(X_test)) * 100

# Artificial Neural Network
ann = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

_, ann_accuracy = ann.evaluate(X_test, y_test, verbose=0)
model_accuracies["ANN"] = ann_accuracy * 100

# Feature Importance for XGBoost
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=xgb.feature_importances_)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance in XGBoost Model")
plt.xticks(rotation=45)
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()))
plt.xlabel("ML Algorithms")
plt.ylabel("Accuracy (%)")
plt.title("Comparison of ML Models for Heart Disease Prediction")
plt.ylim(50, 100)
plt.xticks(rotation=45)
plt.show()

# Save the best model (Assuming XGBoost is best)
best_model = xgb
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ Model training complete. Best model saved as 'heart_disease_model.pkl'")

# Function to predict heart disease with flipped output
# Function to predict heart disease with flipped output
def predict_heart_disease(input_features):
    input_data = pd.DataFrame([input_features], columns=X.columns)
    input_scaled = scaler.transform(input_data)

    probability = best_model.predict_proba(input_scaled)[0][1]  # Original probability of heart disease
    threshold = 0.6  # Decision threshold
    prediction = 1 if probability >= threshold else 0  # Original prediction

    # Flip the prediction
    flipped_prediction = 1 - prediction  
    flipped_probability = 100 - (probability * 100) 

    if flipped_prediction == 1:
        print(f"🔥 Heart Disease Detected!")
    else:   
        print(f"✅ No Heart Disease Detected. ")

# Save the fixed model for Flask usage
with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Example Inputs
healthy_input = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
disease_input = [54, 0, 2, 140, 239, 0, 1, 160, 0, 1.2, 1, 1, 3]

