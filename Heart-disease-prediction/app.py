from flask import Flask, render_template, request
import numpy as np
import pickle

# Load trained model and scaler
with open("heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input (13 features)
        input_data = [float(x) for x in request.form.values()]
        
        # Convert input to NumPy array
        input_data = np.array(input_data).reshape(1, -1)
        
        # Scale input data
        input_data = scaler.transform(input_data)
        
        # Get probability for heart disease
        probability = model.predict_proba(input_data)[0][1]  # Probability of disease
        
        # Decision threshold
        threshold = 0.6
        prediction = 1 if probability >= threshold else 0
        
        # Flip the prediction
        flipped_prediction = 1 - prediction
        flipped_probability = 100 - (probability * 100)  # Flipped probability

        # Generate result text
        if flipped_prediction == 1:
            result = f" Heart Disease Detected!)"
        else:
            result = f" No Heart Disease Detected.)"

        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return render_template("index.html", prediction_text="❌ Error: " + str(e))

if __name__ == "__main__":              
    app.run(debug=True)
