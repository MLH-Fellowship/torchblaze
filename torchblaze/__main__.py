
from __future__ import print_function

import os

import fire
from pyfiglet import figlet_format
from PyInquirer import prompt

from .template import startproject
from .dockerise import createdockerfile, dockerfilechecker, buildimage, runimage
from .apitests import get_routes,tests
from .Utils.utils import style, log


class TorchBlaze(object):
    """The CLI class. The class methods will act as CLI commands.
    """

    def __init__(self):
        pass

    def generate_template(self, project_name):
        """Generates the project template upon running the 
        "deploy generate_template --project_name <name>" command

        Arguments:
            project_name::str- Name of the project

        Returns:
            None
        """
        startproject(project_name)

    def generate_docker(self, image_name):
        """Generates the template Dockerfile and Docker image for the project.

        Arguments:
            image_name::str- Name of the Docker image that is to be generated.
        
        Returns:
            None
        """

        # Checking for Docker file exits or not 
        if not dockerfilechecker():
            # Creating a Docker file 
        	createdockerfile()
        	print('Default Dockerfile created.')
       	else:
            print('A Dockerfile already exists for this project.')
        # Building a Docker Image 
        buildimage(image_name)


    def run_docker(self, image_name):
        """Generates the template Dockerfile and Docker image for the project.

        Arguments:
            image_name::str- Name of the Docker image that is to be generated.
        
        Returns:
            None
        """
        runimage(image_name)


    def api_test(self):
        """A automated API testing used to check whether the API route is working correctly or not

        Arguments:
            None
        
        Returns:
            None
        """
        # Firstly we need to run the file and then start testing
        # Getting the list of routes from the app.py 
        routes=get_routes()
        # print(routes)
        HOST_ENDPOINT="http://127.0.0.1:8080"
        tests(routes,HOST_ENDPOINT)

'''
The following is the implementation of the TorchBlaze instance in functional way

'''

def generate_template():
    """Generates the project template upon running the 
    "deploy generate_template --project_name <name>" command

    Arguments:
        project_name::str- Name of the project

    Returns:
        None
    """
    while True:
        questions = [
            {
                'type': 'input',
                'name': 'project_name',
                'message':'Project Name',
            },
            {
                'type': 'confirm',
                'name': 'Confirm',
                'message': 'Confirm'
            }
        ]    

        answers = prompt(questions, style=style)
        if len(answers['project_name']) == 0: log('Error: Project Name cannot be empty', "red")
        else: 
            project_name = answers['project_name']
            break

    print(project_name)
    startproject(project_name)

def generate_docker():
    """Generates the template Dockerfile and Docker image for the project.

    Arguments:
        image_name::str- Name of the Docker image that is to be generated.
    
    Returns:
        None
    """
    while True:
        questions = [
            {
                'type': 'input',
                'name': 'image_name',
                'message':'Docker Image Name',
            },
            {
                'type': 'confirm',
                'name': 'Confirm',
                'message': 'Confirm'
            }
        ]    

        answers = prompt(questions, style=style)
        if len(answers['image_name']) == 0: log('Error: Image Name cannot be empty', "red")
        else: 
            image_name = answers['image_name']
            break

    # Checking for Docker file exits or not 
    if not dockerfilechecker():
        # Creating a Docker file 
        createdockerfile()
        print('Default Dockerfile created.')
    else:
        print('A Dockerfile already exists for this project.')
    # Building a Docker Image 
    buildimage(image_name)

def run_docker(image_name):
    """Generates the template Dockerfile and Docker image for the project.

    Arguments:
        image_name::str- Name of the Docker image that is to be generated.
    
    Returns:
        None
    """
    runimage(image_name)

def api_test():
    """A automated API testing used to check whether the API route is working correctly or not

    Arguments:
        None
    
    Returns:
        None
    """
    # Firstly we need to run the file and then start testing
    # Getting the list of routes from the app.py 
    routes=get_routes()
    # print(routes)
    HOST_ENDPOINT="http://127.0.0.1:8080"
    tests(routes,HOST_ENDPOINT)


FEATURE2FUNCTION = {
    'Create Project Template':generate_template,
    'Generate Docker Image':generate_docker,
    'Run Docker Image':run_docker,
    'Run API Tests': api_test,
}

def main():
    
    log("TorchBlaze", color="red", figlet=True)
    log(
        "A CLI-based python package that provides a suite of functionalities to perform end-to-end ML using PyTorch.", 
        "green",
    )
    log(
        "Please select one of the options", 
        "green",
    )

    questions = [
        {
            'type':'checkbox',
            'qmark': '😃',
            'message': '',
            'name': 'CLI Options',
            'choices':[
                {
                    'name':'Create Project Template',
                    'value':'Create Project Template',
                },
                {
                    'name':'Generate Docker Image',
                    'value':'Generate Docker Image',
                },
                {
                    'name':'Run Docker Image',
                    'value':'Run Docker Image',
                },
                {
                    'name':'Run API Tests',
                    'value':'Run API Tests',
                },                
                {
                    'name':'Exit',
                    'value':'Exit',
                },                                
            ],
            'validate': lambda answer: 'You must one choose option only.' if (len(answer) == 0 or len(answer)>1) else True
        }
    ]    

    # Handles the following cases
    # 1. Single option chosen
    # 2. Multiple Option Chose
    # 3. No option chosen

    answers = prompt(questions, style=style)
    if not len(answers['CLI Options']): exit

    if len(answers['CLI Options']) >= 1: 
        chosen = answers['CLI Options'][0]

    FEATURE2FUNCTION.get(chosen)()

    # fire.Fire(TorchBlaze)

if __name__ == '__main__':
    main()
