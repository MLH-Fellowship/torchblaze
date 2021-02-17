from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import numpy as np
import traceback
import json
from model import Net
import torch
import cv2

app = Flask(__name__)
api = Api(app)

model = Net()
model.load_state_dict(torch.load('mnist_cnn.pt'))
model.eval()

# convert request_input dict to input accepted by model.
def parse_input(request_input):
    img = request_input.read()
    img = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(28,28))
    img = img.reshape((1,1,28,28))
    return torch.Tensor(img)


# convert model prediction to dict to return as JSON
def parse_prediction(prediction):
    prediction = prediction.argmax(dim=1, keepdim=True)
    return {'class':int(prediction[0][0])}

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

api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
