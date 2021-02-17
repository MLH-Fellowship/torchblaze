#!/usr/bin/python
# -*- coding: utf-8 -*-
import importlib.util
import os
from flask import url_for


def has_no_empty_params(rule):
    defaults = (rule.defaults if rule.defaults is not None else ())
    arguments = (rule.arguments if rule.arguments is not None else ())
    return len(defaults) >= len(arguments)


def get_routes():
    spec = importlib.util.spec_from_file_location('app', 'app.py')
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    routes = []
    for rule in app.app.url_map.iter_rules():
        if 'POST' in rule.methods and has_no_empty_params(rule):
            routes.append((str(rule),rule.endpoint))

    return routes


if __name__ == '__main__':
    get_routes()
