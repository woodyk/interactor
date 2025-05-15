#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: setup.py
# Author: Ms. White 
# Description: 
# Created: 2025-05-12 15:19:57
# Modified: 2025-05-15 00:10:34

import os
from setuptools import setup, find_packages

def load_requirements(filename="requirements.txt"):
    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(here, filename)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-interactor",
    version="0.1.2.2",
    description="Universal AI interaction library with session and tool support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wadih Khairallah",
    author_email="woodyk@gmail.com",
    url="https://github.com/woodyk/interactor",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=load_requirements(),
    entry_points={
        "console_scripts": [
            "interactor=interactor.interactor:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)

