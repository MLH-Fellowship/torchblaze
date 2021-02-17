#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pkg_resources


def createdockerfile(project):
    curr_dir = os.getcwd()
    f = os.path.join(curr_dir, 'Dockerfile')
    with open(f, 'w+') as writefile:
        writefile.write(pkg_resources.resource_string('deploy',
                        'template_files/docker.txt').decode('utf-8'))


def dockerfilechecker():
    curr_dir = os.getcwd()
    rootfiles = os.listdir(curr_dir)
    dockerfilename = 'Dockerfile'
    if dockerfilename in rootfiles:
        return True
    return False


def buildimage(image_name):
    print('Docker Image ' + image_name + ' is getting created')
    os.system('docker build -t ' + image_name + ' .')
    print('Docker Image Build Done')