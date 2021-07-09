# USAGE
# Start the server:
#   python3 run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F file=@input.jpg 'http://localhost:5000/predict'
# Submit a a request via Python:
#   python3 simple_request.py

from flask import Flask, request, jsonify
import logging

from arkham.Bayes.Quantify.evaluate import predict as model_predict
from arkham.utils.model_utils import load_model_path

global models

models = {
    "process_label": load_model_path("/mnt/lerna/models/SROIE2019")
    # NL-UW/062320-101719_NL-UW")
}

# initialize our Flask application and the model
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    label = "process_label"
    content_out = {"success": False, "label": label}
    model = models[label]
    inputted = request.json["document_text"]
    try:
        out = model_predict(*model, inputted)
        content_out["success"] = True
    except Exception as e:
        logging.error(e)
        out = ""
    content_out = {**content_out, **out}
    return jsonify(content_out)


# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print("please wait until server has fully started")
    app.run()
