from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('savedmodel.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Make prediction
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    
    # Redirect to the result page with prediction
    return redirect(url_for('result', prediction=prediction))

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    
    # Set image URL based on the prediction
    image_url = None
    if prediction == 'Iris-virginica':
        image_url = url_for('static', filename='vergincia1')
    elif prediction == 'Iris-setosa':
        image_url = url_for('static', filename='setosa1')
    elif prediction == 'Iris-versicolor':
        image_url = url_for('static', filename='ver')
    
    return render_template('result.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
