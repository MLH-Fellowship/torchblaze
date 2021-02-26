---
id: setup
title: Setup a TorchBlaze ML Project
---

TorchBlaze allows you to setup your ML project within no time! With our template, you will be ready to jump in to making your model and training it without worrying about the hassles of testing and deployment, and leaving that upto us. The template is just to display a basic example to get you started on your journey with TorchBlaze and build some great things!

## Getting Started with the template
---

Generate the template for your project using the following command:

```shell
foo@bar:~$ torchblaze generate_template <your_project_name>
```
This will create a directory titled `<your_project_name>` and contain the template files.

:::note

Before proceeding, make sure you have torch installed on your system/virtual environment. In case you do not have it installed, you can install your relevant PyTorch version from [here](https://pytorch.org/get-started/locally/).  

:::

Install the required python packages for running the template example using the following command:

```shell
foo@bar:~$ pip install -r requirements.txt
```

You are now set to modify the template to your liking and continue with the deployment!

## Running the template

To begin training the existing model definition present in `model/model.py` and save the final model, you can execute:

```shell
foo@bar:~$ python model/train.py --save-model
```

You can supply any of the command-line arguments defined in `model/train.py` to enhance the model training in several ways. Once your model is trained and saved, you can either execute your ready-to-deploy flask-api on your local machine like this:

```shell
foo@bar:~$ python app.py
```

or you could generate a docker image to wrap your flask-api and model with the command:

```shell
foo@bar:~$ torchblaze generate_docker <your_docker_image_name> 
```

Once you are convinced your API is ready for deployment, you can use the `Procfile` that has been generated to deploy your API as a Heroku App.

## Contents of the Template
---

### # model/model.py

#### File Purpose

This file should contain the:

1. model layers definition
2. model architecture definition

#### File Contents:

1. Model definition defined in `__init__(self)` which takes as input a batch of 28x28 grayscale images and outputs a batch of softmax-probability vectors, each of length 10.
2. CNN based model for classifying handwritten numbers with the model architecture defined in `forward()`. 

---

### # model/train.py

#### File Purpose

This file should contain the functionality for:

1. setting hyperparameters through methods such as command-line arguments or config files.
2. Loading and Pre-processing data for model input
3. Creating model, optimizer and criterion. (optional: scheduler)
4. Training Loop to iterate once over the complete training dataset and train the model.
5. Testing Loop to perform inference once over the complete testing dataset.
6. Epoch Loop to call the training loop a certain number of times as defined in hyperparameters
7. Evaluate model on test set to infer the model accuracy
8. Ability to save the model to disk.

#### File Contents:

1. Hyperparameter setup using command-line arguments
2. Loading MNIST dataset using `torchvision.Datasets` and storing it in a Dataloader.
3. Creating model from `model/model.py` and using `Adadelta` optimizer and NLL Loss.
4. Training Loop in `train` function.
5. Unit Testing using `torchblaze.mltests` which is clearly detailed [here](https://mlh-fellowship.github.io/torchblaze/docs/mltests)
6. Testing Loop in `test` function.
7. Epoch Loop, model evaluation and model saving in `main` function.

### # app.py

#### File Purpose

This file should contain the functionality for:

1. creating flask-restful API from flask App.
2. listen for POST or GET request made by a client.
3. parse input from request to prepare as input to the model.
4. parse the model output to return the model inference in a user-friendly manner.

#### File Contents:

1. Flask and Flask-RESTful packages are used to create the REST API.
2. A `Resource` is created which listens to POST requests for image files.
3. The Resource is added to the API along with two other dummy resources for running [API tests](https://mlh-fellowship.github.io/torchblaze/docs/apitest)
4. Input image file is resized and converted to a `torch.Tensor`, prepared for inference.
5. The argmax of model output gives the predicted class which is returned as a JSON to the client. 

### # model/utils.py

#### File Purpose

This file should contain the functionality for any additional helper functions that are required for the other sections of the model code but do not fit into the core ML workflow as defined in the purpose of `model/train.py`.


### # Procfile

Heroku Apps require a Procfile which specifies the commands that would run on the start of a Heroku App. A more detailed explanation regarding Procfiles can be found [here](https://devcenter.heroku.com/articles/procfile). The template provides a Procfile that you can directly use to deploy your API.

### # Docker

This file is used to create the docker image for the API to make it runnable on any system. The Docker image generation has been described in detail [here](https://mlh-fellowship.github.io/torchblaze/docs/docker).

### # tests.json

This file is used to create the dummy requests which are used by APITests to ensure the working of the API. The format and usage of the `tests.json` file has been described in detail [here](https://mlh-fellowship.github.io/torchblaze/docs/apitest)
