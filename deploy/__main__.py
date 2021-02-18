import fire
from .template import startproject
from .dockerise import createdockerfile, dockerfilechecker, buildimage
from .apitests import get_routes
import os

def main():
  fire.Fire(Deploy)

class Deploy(object):
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

    def api_test(self):
        """A automated API testing used to check whether the API route is working correctly or not

        Arguments:
            None
        
        Returns:
            None
        """
        # Getting the list of routes from the app.py 
        routes=get_routes()
        print(routes)



if __name__ == '__main__':
    main()
