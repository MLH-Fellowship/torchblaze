import importlib
import os
from flask import url_for
import sys
import requests
import json
# Setting the System Path to current directory

curr_dir = os.getcwd()
sys.path.append(curr_dir)



def has_no_empty_params(rule):
    """Check whether the route rules contain any empty parameters or not

    Arguments:
        input takes the route which we want to check 
        
    Returns:
        Bool: True if the route doesn't contain any empty parameters, otherwise false.
    """
    
    defaults = (rule.defaults if rule.defaults is not None else ())
    arguments = (rule.arguments if rule.arguments is not None else ())
    return len(defaults) >= len(arguments)


def get_routes():
    """Get all the list of routes with the functions initialised in the app.py file

    Arguments:
        None

    Returns:
        list: returns the list of routes defined in app.py
    """
    
    # Loading the app.py file as a module using library importlib

    from app import app
    routes = []
    for rule in app.url_map.iter_rules():
        # Checking whether the rule has any empty params

        if has_no_empty_params(rule):
            # The list contains the tuple which comprises of route method,route path,route end point

            routes.append((rule.methods,str(rule),rule.endpoint))
    return routes


def tests(routes,baseurl):
    """It sends the request to the routes by taking test cases from tests.json"

    Arguments:
        input takes list of routes and the baseurl of the api

    Returns:
        None
    """
    # Loading tests.json file
    curr_dir=os.getcwd()
    f = os.path.join(curr_dir, "tests.json")
    with open(f, "r") as jsonfile:
        data=json.load(jsonfile)
        #print(data)
    for i in routes:
        
        # GET Method Testing
        if 'GET' in i[0]:
            #print("get")
            #print(baseurl)
            response=requests.get(baseurl+str(i[1]))
            status_code_get=response.status_code
            if(status_code_get==200):
                print("route",str(i[1]),"get successful")
            else:
                print(i[1],"failed with return status_code",status_code_get)
        # POST Method Implemntation
        elif 'POST' in i[0]:
            endpoint=str(i[2])
            if endpoint=='makeprediction':
                continue
            end_test_data=data[endpoint]
            for test in end_test_data:
                response=requests.post(baseurl+str(i[1]),json=test)
            status_code_post=response.status_code
            if(status_code_post==200):
                print("route",str(i[1]),"post successful")
            else:
                print(i[1],"failed with return status_code",status_code_post)





if __name__ == '__main__':
    get_routes()
