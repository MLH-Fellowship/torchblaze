import fire
from template import startproject
from dockerise import createdockerfile

class Deploy(object):

  def __init__(self):
      pass

  def generate_template(self, project_name):
    startproject(project_name)

  def generate_docker(self,project_name):
  	createdockerfile(project_name)
    

if __name__ == '__main__':
  fire.Fire(Deploy)