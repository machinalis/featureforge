# coding: utf-8

import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


base_path = os.path.dirname(os.path.abspath(__file__))
long_description = open(os.path.join(base_path, 'README.md')).read()

setup(
    name="featureforge",
    version="0.1",
    description="A library to build and test machine learning features",
    long_description=long_description,
    author="Rafael Carrascosa, Daniel Moisset, Javier Mansilla",
    packages=[
        "featureforge",
    ],
    install_requires=[
        "mock",
        "schema",
        "numpy",
        "scipy"
    ],
    include_package_data=False,
)
