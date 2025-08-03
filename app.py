import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Convert to NumPy array and reshape
    values = np.array(list(data.values())).reshape(1, -1)
    print(values)

    # Scale the input
    new_data = scaler.transform(values)

    # Predict
    prediction = model.predict(new_data)
    print(prediction)

    return jsonify(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
