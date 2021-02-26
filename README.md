![TorchBlaze](./documentation/static/img/torchblaze.svg)

# TorchBlaze 
[Link to Documentation](https://mlh-fellowship.github.io/torchblaze/)
---

A CLI-based python package that provides a suite of functionalities to perform end-to-end ML using PyTorch. 

### The following are the set of functionalities provided by the tool:
---

* __Flask-API Template__: Set up the basic PyTorch project sturcture and an easily tweakable flask-RESTful API with a single CLI command. Deploying your ML models has never been so easy.

* __Test ML API__: Once you have set up your API, test all the API end-points to ensure you get the expected results before pushing your API to deployment.

* __Dockerizing__: A simplified, single-command, easy dockerization for your ML API.  

* __ML Model Test Suite__: The package comes with a built-in test suite that evaluates your PyTorch models over a set of tests to look for any errors that otherwise might not be traceable easily.

### Here are the available list of commands:
---

* Setting-up the Template Project:

```console
foo@bar:~$ torchblaze generate_template --project_name example
```

* Building Docker Image (Requires Docker Installed):
> First cd to the root project directory containing app.py file.

```console
foo@bar:~$ torchblaze generate_docker --image_name example_image
```

* Run Docker Image (Requires Docker Installed):

```console
foo@bar:~$ torchblaze run_docker --image_name example
```

* Performing API Tests:

> First cd to the root project directory containing app.py file.
```console
foo@bar:~$ torchblaze api_tests
```

* Performing Model Testing:


> Import the mltests package
```py
import torchblaze.mltests as mls
```
> Then use the variety of testing methods available in the mltests package. Run the following command to get the list of available methods.
```py
dir(mls)
```
> To check the documentation for any of the available tests, use the help method:
```py
help(mls.<method_name>)
```
