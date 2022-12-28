import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__) #Initialize the app
model = pickle.load(open('model.pkl', 'rb')) # loading the model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    initial_vals = [float(x) for x in request.form.values()]
    final_vals = [np.array(initial_vals)]

    prediction = model.predict(final_vals) #prediction
    return render_template('index.html', prediction_text='Predicted class is: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)