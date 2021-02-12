import os

def createdockerfile(project:str):
    curr_dir = os.getcwd()
    root_dir = os.path.join(curr_dir, project)

    # print(curr_dir)

    f = os.path.join(root_dir, 'Dockerfile')
    readfile = open('./deploy/template_files/' + 'docker.txt', 'r')
    with open(f, 'w+') as dockerfile:
        for line in readfile:
            dockerfile.write(line)


def dockerfilechecker(project):
    curr_dir = os.getcwd()
    root_dir = os.path.join(curr_dir, project)
    rootfiles = os.listdir(root_dir)
    dockerfilename = 'Dockerfile'
    if dockerfilename in rootfiles:
        return True
    return False
