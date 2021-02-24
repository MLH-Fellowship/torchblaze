import os
import pkg_resources
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)


def createdockerfile():
    """Creates a template Dockerfile for the project.
    
    Arguments:
        None
    
    Returns:
        None
    """
    curr_dir = os.getcwd()
    f = os.path.join(curr_dir, 'Dockerfile')
    with open(f, 'w+') as writefile:
        writefile.write(pkg_resources.resource_string('torchblaze',
                        'template_files/docker.txt').decode('utf-8'))


def dockerfilechecker():
    """Checks if a Dockerfile already exists in the project directory.

    Arguments:
        None
    Returns:
        Bool: True if the Dockerfile already exists, otherwise false. 
    """
    curr_dir = os.getcwd()
    rootfiles = os.listdir(curr_dir)
    dockerfilename = 'Dockerfile'
    if dockerfilename in rootfiles:
        return True
    return False


def buildimage(image_name:str):
    """Creates the Docker image for the API/project.

    Arguments:
        image_name::str- Name for the Docker image
    
    Returns:
        None
    """
    print('Docker Image ' + image_name + ' is getting created')
    os.system('docker build -t ' + image_name + ' .')
    print('Docker image build completed')


def runimage(image_name:str):
    """Runs the Docker Image Container

    Arguments:
        image_name::str- Name for the Docker image
    
    Returns:
        None
    """
    print("Docker Image Running")
    os.system("docker run -p 8080:8080 "+image_name)
    