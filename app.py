from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('tip_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    total_bill = float(request.form['total_bill'])
    size = int(request.form['size'])
    input_data = np.array([[total_bill, size]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)