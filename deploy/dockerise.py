#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import pkg_resources


def createdockerfile(project):
    curr_dir = os.getcwd()
    root_dir = os.path.join(curr_dir, project)

    # print(curr_dir)

    f = os.path.join(root_dir, 'Dockerfile')
    with open(f, 'w+') as writefile:
        writefile.write(pkg_resources.resource_string('deploy',
                        'template_files/docker.txt').decode('utf-8'))


def dockerfilechecker(project):
    curr_dir = os.getcwd()
    root_dir = os.path.join(curr_dir, project)
    rootfiles = os.listdir(root_dir)
    dockerfilename = 'Dockerfile'
    if dockerfilename in rootfiles:
        return True
    return False
