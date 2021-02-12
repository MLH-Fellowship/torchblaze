#!/usr/bin/python
# -*- coding: utf-8 -*-
import fire
from .template import startproject
# from dockerise import createdockerfile, dockerfilechecker
# import os

def main():
  fire.Fire(Deploy)

class Deploy(object):

    def __init__(self):
        pass

    def generate_template(self, project_name):
        startproject(project_name)

    # def generate_docker(self, project_name):
    #     if not dockerfilechecker(project_name):
    #     	createdockerfile(project_name)
    #     	print('Default Docker File Created')
    #    	else:
    #         print('Dockerfile already present')


if __name__ == '__main__':
    main()
