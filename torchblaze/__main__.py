import fire
from .template import startproject
from .dockerise import createdockerfile, dockerfilechecker, buildimage, runimage
from .apitests import get_routes,tests
import os

def main():
  fire.Fire(TorchBlaze)

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



if __name__ == '__main__':
    main()
