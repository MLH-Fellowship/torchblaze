from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import numpy as np
import traceback
import json
from model.model import Net
import torch
import cv2

app = Flask(__name__)
api = Api(app)

model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()

# convert request_input dict to input accepted by model.
def parse_input(request_input):
    """parse input to make it ready for model input.

    Arguments:
        request_input::file- input received from API call.

    Returns:
        model_ready_input::torch.Tensor- data ready to be fed into model.
    """
    img = request_input.read()
    img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    img = img.reshape((1,1,28,28))
    return torch.Tensor(img)


# convert model prediction to dict to return as JSON
def parse_prediction(prediction):
    """parse prediction to prepare a request response.

    Arguments:
        prediction::torch.Tensor- output of model inference

    Returns:
        response::dict- dictionary to return as a JSON response.
    """
    prediction = prediction.argmax(dim=1, keepdim=True)
    return {'class':int(prediction[0][0])}

# class inheriting from Resource prepares it to become an API endpoint
class MakePrediction(Resource):
    @staticmethod
    def post():
        if model:
            try:
                request_input = request.files['file']

                model_input = parse_input(request_input)

                prediction = model(model_input)

                model_output = parse_prediction(prediction)

                return jsonify(model_output)

            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'trace': 'No model found'})

api.add_resource(MakePrediction, '/predict') # add endpoint to API


class dummy_post(Resource):
    @staticmethod
    def post():
        if model:
            try:
                request_input = request.get_json()

                return jsonify(request_input)

            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'trace': 'No model found'})

api.add_resource(dummy_post, '/dummy_post')


class dummy_get(Resource):
    @staticmethod
    def get():
        if model:
            try:
                return str("dummy get request")

            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return jsonify({'trace': 'No model found'})

api.add_resource(dummy_get, '/dummy_get')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)
