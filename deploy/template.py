import os
import pkg_resources

def startproject(project: str):
    # try: 
        curr_dir = os.getcwd()
        root_dir = os.path.join(curr_dir, project)
        
        # creating the main project directory
        os.mkdir(root_dir) 

        # creating the files in root project directory
        f = os.path.join(root_dir, ".gitignore")
        with open(f, "w+") as gitignore:
            gitignore.write(" ")
        
        f = os.path.join(root_dir, "README.md")
        with open(f, "w+") as readme:
            readme.write("# "+ project+"\n---")
        
        f = os.path.join(root_dir, "app.py")
        with open(f, "w+") as writefile:
            writefile.writelines(pkg_resources.resource_string('deploy', 'template_files/app.py').decode('utf-8').split('\n'))
        
        # creating the model directory and sub-dir/files
        model_dir = os.path.join(root_dir, 'model')
        os.mkdir(model_dir)
        os.mkdir(os.path.join(model_dir, 'data'))
        model_files = ['utils', 'model', 'train']
        
        for file in model_files:
            f = os.path.join(model_dir, file+'.py')
            with open(f, "w+") as writefile:
                    writefile.writelines(pkg_resources.resource_string('deploy', f'template_files/{file}.py').decode('utf-8').split('\n'))
    # except:
    #     print("A directory with project name already exists. Kindly choose a different name.")
    

if __name__ == "__main__":
    startproject('test_project')
