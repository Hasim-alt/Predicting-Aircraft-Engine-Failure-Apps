from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

model = tf.keras.models.load_model('engine_model.h5')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl') # List of strings

@app.route('/')
def home():
    # Pass the feature list to HTML so we can dynamically generate inputs or labels
    return render_template('index.html', features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feature_name in selected_features:
            val = request.form.get(feature_name)
            if val is None:
                return f"Error: Missing value for {feature_name}"
            input_data.append(float(val))

        features_array = np.array([input_data])
        scaled_features = scaler.transform(features_array)
        prediction_prob = model.predict(scaled_features)[0][0]
        
        # Threshold Logic (>= 0.5 Failure)
        status = "FAILURE PREDICTED" if prediction_prob >= 0.5 else "Engine Normal"
        color = "red" if prediction_prob >= 0.5 else "green"

        return render_template('index.html', 
                               features=selected_features,
                               prediction_text=f'Status: {status}', 
                               probability=f'Failure Probability: {prediction_prob:.4f}',
                               color=color)

    except Exception as e:
        return render_template('index.html', features=selected_features, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)