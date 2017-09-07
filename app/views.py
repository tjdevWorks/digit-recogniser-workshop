from flask import render_template, flash, redirect, request, jsonify
from app import app
from app.model.preprocessor import preprocess
from app.model.tensorflow_predict import predict
import json
import sys

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	app.logger.debug("Went to OCR page")
	return render_template('index.html', title='Optical Character Recognition', prediction=None)


@app.route('/_do_ocr', methods=['GET', 'POST'])
def do_ocr():
    """Add two numbers server side, ridiculous but well..."""
    app.logger.debug("Accessed _do_ocr page with image data")
    data = request.args.get('imgURI', 0, type=str)
    index = request.args.get('index', 0, type=int)
    #char_prediction, percentage = predict(preprocess(data))
    #result = "Predicting you entered a: {} with {:.2f} %".format(char_prediction, percentage)
    result = "You still have to build a model to view results"
    #app.logger.debug("Recognized a character "+str(char_prediction))
    return jsonify(result=result)
