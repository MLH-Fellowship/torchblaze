from setuptools import setup, find_packages
from io import open
from os import path

import pathlib
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# automatically captured required modules for install_requires in requirements.txt
with open(path.join(HERE, 'requirements.txt'), 'r') as f:
    all_reqs = f.read().split('\n')

print(all_reqs)

install_requires = [x.strip() for x in all_reqs if ('git+' not in x) and (
    not x.startswith('#')) and (not x.startswith('-'))]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs \
                    if 'git+' not in x]
setup (
 name = 'torchblaze',
 description = 'A CLI-based python package that provides a suite of functionalities to perform end-to-end ML using PyTorch.',
 version = '1.0.4',
 packages = find_packages(), # list of all packages
 install_requires = install_requires,
 python_requires='>=3.7.0',
 include_package_data = True,
 entry_points='''
        [console_scripts]
        torchblaze=torchblaze.__main__:main
    ''',
 author="Sai Durga Kamesh Kota",
 long_description=README,
 long_description_content_type="text/markdown",
 license='MIT',
 url='https://github.com/MLH-Fellowship/torchblaze',
  author_email='ksdkamesh99@gmail.com',
  classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ]
)
