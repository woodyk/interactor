#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: setup.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-05-05 16:54:44
# Modified: 2025-05-05 17:58:10

from setuptools import setup, find_packages

def load_requirements(path="requirements.txt"):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="interactor",
    version="0.1.0",
    description="Universal AI interaction library with session and tool support",
    author="Wadih Khairallah",
    author_email="woodyk@gmail.com",
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
        "Operating System :: OS Independent",
    ],
)
