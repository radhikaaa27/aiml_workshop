from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)


# Load trained model
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():

    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    input_data = np.array([[
        float(data['Pregnancies']),
        float(data['Glucose']),
        float(data['BloodPressure']),
        float(data['SkinThickness']),
        float(data['Insulin']),
        float(data['BMI']),
        float(data['DiabetesPedigreeFunction']),
        float(data['Age'])
    ]])
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': 'Diabetic' if prediction == 1 else 'Not Diabetic'})

if __name__ == '__main__':
    app.run(debug=True)
