---
id: installation
title: Installation
---
 
TorchBlaze is a pip-installable command line tool and Python package, that provides a suite of functionalities to perform end-to-end ML using PyTorch. TorchBlaze was designed to fast-track things when it comes to deploying ML models online, thus allowing the developer to focus on what matters the most— Ensuring that their models give a high inference accuracy.

Once that is done, you can use TorchBlaze's easily tweakable templates to quickly create APIs for your ML models. Whether you want to deploy on __Heroku__ or create a __Docker Image__, everything is just a few commands away! 

### The following are the set of functionalities provided by the TorchBlaze:
---

* __Flask-API Template__: Set up the basic PyTorch project sturcture and an easily tweakable flask-RESTful API with a single CLI command. Deploying your ML models has never been so easy.

* __ML Model Test Suite__: The package comes with a built-in test suite that evaluates your PyTorch models over a series of tests that look for any errors that otherwise might not be traceable easily.

* __Test ML API__: Once you have set up your API, use the automated tests to test all the API end-points, ensuring that you get the expected results, before pushing your API to deployment.

* __Dockerizing__: A simplified, single-command, easy dockerization for your ML API.  


## Getting Started with TorchBlaze
---

### Installing TorchBlaze

Install TorchBlaze from PyPI using the following command:

```shell
foo@bar:~$ pip install torchblaze
```
This will install the latest version of torchblaze on your system. 

:::note

Dockerizing the project, or deployment to Heroku requires Docker and Heroku CLI installed on your system.  

:::

### Installing External Dependencies

To install Docker on your system, you can follow these guides:

Install Docker Engine on Linux: [Official Documentation](https://docs.docker.com/engine/install/)

Install Docker Desktop on Windows: [Official Documentation](https://docs.docker.com/docker-for-windows/install/)

Install Docker Desktop on Mac: [Official Documentation](https://docs.docker.com/docker-for-mac/install/)


### Installing Heroku CLI

For deploying your ML API on Heroku, you need to install the __Heroku CLI__. Note that the required Heroku Procfile gets generated at the time of generating a project template using TorchBlaze.

Install Heroku CLI: [Official Documentation](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)

