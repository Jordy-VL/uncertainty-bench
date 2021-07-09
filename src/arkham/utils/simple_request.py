# python simple_request.py INPUT

import requests
import sys
import json

# initialize the Keras REST API endpoint URL along with the input
KERAS_REST_API_URL = "http://localhost:5000/predict"
print(sys.argv)
inputted = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "ik wil graag mijn autopolis wijzigen. Ik ben omnium beu, geef mij maar dringend BA. Alvast bedankt, Jos"
)

# construct the payload for the request
payload = {"document_text": inputted}
print(payload)

# submit the request
r = requests.post(KERAS_REST_API_URL, json=payload).json()

# ensure the request was sucessful
if r["success"]:
    print(json.dumps(r, indent=4))
    label = r["label"]
    r = r["simple"]
    print("{}: {}. {:.4f}".format(label, r["text"], r["confidence"]))
else:
    print("Request failed")