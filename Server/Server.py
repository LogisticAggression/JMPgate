"""
Listens on the specified port for incoming sequences and does things with them
"""
import json
import base64
import numpy as np
from Vectorizers import ChainerRNNVectorizer
from flask import Flask, request, jsonify

app = Flask(__name__)

class InvalidUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/jmpgate', methods=['POST'])
def JMPgate():
    """
    Receives requests containing JMPgate taskings and executes them.  A properly formed tasking should have
    the following elements in a json blob:
    task - The backend module to run after vectorization
    buffer - The sequence of bytes to be vectorized

    :return: The response from the backend task
    """
    try:
        req_data = request.get_json()
    except:
        raise InvalidUsage('Could not parse request', status_code=400)
    if 'buffer' not in req_data:
        raise InvalidUsage('Must send something to vectorize', status_code=400)
    if 'task' not in req_data:
        raise InvalidUsage('Must send a task', status_code=400)
    vector = vectorizer.vectorize(req_data['buffer'])
    return "Got stuff"



if __name__ == "__main__":
    vectorizer = ChainerRNNVectorizer("model_name")
    app.run(debug=True, port=6666)
