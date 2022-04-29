# Load libraries
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model

import flask
from flask import Flask, jsonify, request
from datetime import date

# initialize our Flask application
app = Flask(__name__)

# load the model, and pass in the custom metric function
global graph
graph = tf.compat.v1.get_default_graph()
model = load_model('model.h5')

@app.route("/output", methods=["GET"])
def hello():
    today = date.today()
    html=f"<HTTML> <BODY> <STRONG> Today's date is: {today} </STRONG> </BODY> </HTML>"
    return "<HTTML> <BODY> <STRONG> Rock, Paper and Scissors image classification server. <br> Ria Sharma <br> </STRONG> </BODY> </HTML>" + html

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}

    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # if parameters are found, return a prediction
    if (params != None):
        x=pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data["prediction"] = str(model.predict(x)[0][0])
            data["success"] = True

    # return a response in json format 
    return flask.jsonify(data)

#  main thread of execution to start the server
if __name__=='__main__':
    app.run(debug=True)