# End-to-End ML

Your basic deployable machine learning model template has been generated. This template comes with 5 files that you may edit/remove for your project.

1. `model/model.py` - This file is for your model definition (i.e. the class where your model architecture and flow is defined)

2. `model/train.py` - This file is for loading the training data, running the training loops for your model, and finally saving your model.

3. `model/utils.py` - Additional helper functions that may assist you for any processing in the `train.py` file may be placed here.

4. `app.py` - This file defines a Flask RESTful API which is wrapped around your final model and can be deployed to any server for directly running inference.

5. `Procfile` - This file creates the required procfile for quick deployment to Heroku.

### Running base example

1. Create a virtual environment
2. Run `pip install requirements.txt`
3. Run `python model/train.py`
4. Run `python app.py`

These simple steps have set up your localhost to accept images sent through post requests and run a simple classifier that detects the digit present in the image.
