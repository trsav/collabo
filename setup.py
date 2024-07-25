from setuptools import setup, find_packages
setup(
name='collabo',
version='0.1.0',
author='Tom Savage',
author_email='tom.savage@hotmail.com',
description='A package for human-algorithm collaborative Bayesian optimization.',
packages=find_packages(),
classifiers=[
'Programming Language :: Python :: 3',
'License :: OSI Approved :: MIT License',
'Operating System :: OS Independent',
],
python_requires='>=3.6',
)