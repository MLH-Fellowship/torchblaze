#!/usr/bin/python
# -*- coding: utf-8 -*-
import fire
from .template import startproject
from .dockerise import createdockerfile, dockerfilechecker, buildimage
from .apitests import get_routes
import os

def main():
  fire.Fire(Deploy)

class Deploy(object):

    def __init__(self):
        pass

    def generate_template(self, project_name):
        startproject(project_name)

    def generate_docker(self, project_name, image_name):
        if not dockerfilechecker():
        	createdockerfile(project_name)
        	print('Default Dockerfile created.')
       	else:
            print('Dockerfile already present.')
        buildimage(image_name)

    def api_test(self):
        routes=get_routes()
        print(routes)



if __name__ == '__main__':
    main()
