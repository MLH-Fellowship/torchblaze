import os
import pkg_resources

def startproject(project: str):
    """Generates a template project structure for an ML API project.

    Arguments:
        project::str- Project name
    
    Returns:
        None
    """
    try: 
        curr_dir = os.getcwd()
        root_dir = os.path.join(curr_dir, project)
        
        # creating the main project directory
        os.mkdir(root_dir) 

        
        # creating the files in root project directory
        f = os.path.join(root_dir, ".gitignore")  # path to the gitignore file to be created
        with open(f, "w+") as gitignore:  # creating the .gitignore file 
            gitignore.write(" ")  # writing to the gitignore file
        
        
        # same creation logic as the .gitignore file
        f = os.path.join(root_dir, "README.md")
        with open(f, "w+") as writefile:
            writefile.writelines(pkg_resources.resource_string('torchblaze', 'template_files/README.txt').decode('utf-8').split('\n'))

            
        # same creation logic as the .gitignore file
        f = os.path.join(root_dir, "requirements.txt")
        with open(f, "w+") as writefile:
            writefile.writelines(pkg_resources.resource_string('torchblaze', 'template_files/requirements.txt').decode('utf-8').split('\n'))
        
        
        # same creation logic as the .gitignore file
        f = os.path.join(root_dir, "app.py")
        with open(f, "w+") as writefile:
            writefile.writelines(pkg_resources.resource_string('torchblaze', 'template_files/app.py').decode('utf-8').split('\n'))
        
        
        # same creation logic as the .gitignore file
        f = os.path.join(root_dir, "tests.json")
        with open(f, "w+") as writefile:
            writefile.write(pkg_resources.resource_string('torchblaze', 'template_files/tests.txt').decode('utf-8'))

            
        # same creation logic as the .gitignore file
        f = os.path.join(root_dir, "Procfile")
        with open(f, "w+") as writefile:
            writefile.writelines(pkg_resources.resource_string('torchblaze', 'template_files/procfile.txt').decode('utf-8').split('\n'))
        
        # creating the model directory
        model_dir = os.path.join(root_dir, 'model')
        os.mkdir(model_dir)
        
        # crating data sub-directory in the model directory
        os.mkdir(os.path.join(model_dir, 'data'))
        
        
        # creating template files in model directory
        model_files = ['utils', 'model', 'train']
        
        for file in model_files:
            f = os.path.join(model_dir, file+'.py')
            with open(f, "w+") as writefile:
                    writefile.writelines(pkg_resources.resource_string('torchblaze', f'template_files/{file}.py').decode('utf-8').split('\n'))
    except:
        print(f"The directory '{project}' already exists. Kindly choose a different project name.")
    

if __name__ == "__main__":
    startproject('test_project')
