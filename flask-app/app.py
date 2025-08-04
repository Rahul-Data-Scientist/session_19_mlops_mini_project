from flask import Flask, render_template, request
import mlflow
import dagshub
import pickle
from preprocessing_utility import normalize_text

mlflow.set_tracking_uri("https://dagshub.com/Rahul-Data-Scientist/session_19_mlops_mini_project.mlflow")
dagshub.init(repo_owner='Rahul-Data-Scientist', repo_name='session_19_mlops_mini_project', mlflow=True)

app = Flask(__name__)

# load model from registry
model_name = 'my_model'
model_version = 2  # Yahaan hum production wale model ka version daalte hain and hum yeh kaam manually nahi karte hain like this. Instead, we call a function to get the version of model at production stage. Aur agar production stage pe multiple models hain toh doosre tools mein yeh option rehta hai ki kaun sa model choose karna hai. Shayad mlflow mein aisa nahi hai. Ho bhi sakta hai. Check kar lena.

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)

vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', result = None)

@app.route('/predict', methods = ['POST'])
def predict():
    text = request.form['text']

    # clean
    text = normalize_text(text)

    # bow
    features = vectorizer.transform([text])

    # prediction
    result = model.predict(features)

    # show
    return render_template('index.html', result = result[0])

app.run(debug = True)
