"""import pandas as pd
data = pd.read_csv("C:/Users/ASUS/Downloads/Heart-Disease-Prediction-using-Machine-Learning-master/Heart-Disease-Prediction-using-Machine-Learning-master/heart.csv")  # Load dataset
print(data["target"].value_counts())  # Check class balance
"""
probs = model.predict_proba([[70, 1, 1, 170, 300, 1, 2, 100, 1, 3.5, 2, 3, 3]])  
print("Predicted Probabilities:", probs)
