from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
with open('alzheimer_Disease_RF.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form data
            input_data = [
                float(request.form['Age']),
                float(request.form['BMI']),
                int(request.form['Smoking']),
                float(request.form['AlcoholConsumption']),
                float(request.form['PhysicalActivity']),
                float(request.form['DietQuality']),
                float(request.form['SleepQuality']),
                int(request.form['FamilyHistoryAlzheimers']),
                int(request.form['CardiovascularDisease']),
                int(request.form['Diabetes']),
                int(request.form['Depression']),
                int(request.form['HeadInjury']),
                int(request.form['Hypertension']),
                int(request.form['MemoryComplaints']),
                int(request.form['BehavioralProblems']),
                int(request.form['Confusion']),
                int(request.form['Disorientation']),
                int(request.form['PersonalityChanges']),
                int(request.form['DifficultyCompletingTasks']),
                int(request.form['Forgetfulness'])
            ]

            # Convert the input data to a numpy array and reshape for the model
            input_array = np.array([input_data])

            # Predict using the model
            prediction = model.predict(input_array)

            # Convert the prediction to a JSON response
            return jsonify({'prediction': prediction.tolist()})

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return "Invalid request method", 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
