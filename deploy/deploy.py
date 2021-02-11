import fire
from template import startproject

class Deploy(object):

  def __init__(self):
      pass

  def generate_template(self, project_name):
    startproject(project_name)

  def generate_docker(self, x, y):
    pass

if __name__ == '__main__':
  fire.Fire(Deploy)