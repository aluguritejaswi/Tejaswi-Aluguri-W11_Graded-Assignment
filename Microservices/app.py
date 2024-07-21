# Local imports
import datetime

# Third part imports
from flask import request
import pandas as pd

from ms import app
from ms.functions import get_model_response

#Import accuracy value from train.py
from code_model_training.train import Accuracy 



model_name = "Breast Cancer Wisconsin (Diagnostic)"
model_file = 'model/model_binary.dat.gz'
version = "v1.0.0"


@app.route('/info', methods=['GET'])
def info():
    """Return model information, version, how to call"""
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result


@app.route('/health', methods=['GET'])
def health():
    """Return service health"""
    return 'ok'


# @app.route('/predict', methods=['POST'])
# def predict():
#     feature_dict = request.get_json()
#     if not feature_dict:
#         return {
#             'error': 'Body is empty.'
#         }, 500

#     try:
#         # response = get_model_response(feature_dict)
#         response = Accuracy
#     except ValueError as e:
#         return {'error': str(e).split('\n')[-1].strip()}, 500

#     return response, 200

@app.route('/predict', methods=['POST'])
def predict():
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
        }, 500

    try:
        # Assuming get_model_response returns a dictionary with prediction result
        # response = get_model_response(feature_dict)
        
        # If you want to return the accuracy value instead:
        response = {'accuracy': Accuracy}
        
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500

    return response, 200



if __name__ == '__main__':
    app.run(host='0.0.0.0')
