#!/usr/bin/env bash
#
# File: release.sh
# Author: Wadih Khairallah
# Description: 
# Created: 2025-05-12 15:39:52
# Modified: 2025-05-12 16:01:35
#!/bin/bash

set -e  # Exit on first error
set -o pipefail

# ---- Config ----
PACKAGE="ai-interactor"
VERSION="0.1.0"
USE_TESTPYPI=false  # Set to true for test.pypi.org

# ---- Step 0: Setup Environment ----
echo "Setting up virtual environment..."
rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate
pip3 install build twine


# ---- Step 1: Clean old builds ----
echo "Cleaning old build artifacts..."
rm -rf build dist *.egg-info

# ---- Step 2: Optional check for files ----
echo "Checking required files..."
for file in setup.py README.md requirements.txt pyproject.toml; do
  if [[ ! -f "$file" ]]; then
    echo "Missing required file: $file"
    exit 1
  fi
done

# ----Step 3: Build package ----
echo "Building source and wheel distributions..."
python3 -m build

# ---- Step 4: Show contents ----
echo "Contents of dist/:"
ls -lh dist

# ---- Step 5: Upload to PyPI ----
if $USE_TESTPYPI; then
  echo "Uploading to TestPyPI..."
  python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
else
  echo "Uploading to PyPI..."
  python3 -m twine upload dist/*
fi

# ---- Step 6: Verification (manual) ----
echo "Done. To verify:"
echo "    pip install $PACKAGE"

