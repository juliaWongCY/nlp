#!/usr/bin/env bash

# Pre-requisites:
# 1. Python environment installed at ./venv
# 2. Script executed from project repository root directory

if [ ! -d "./venv" ]; then
  echo "Python venv not detected in (proj_root)/venv"
  exit 1
fi

# Activate virtual environment
source ./venv/bin/activate

cd tmp2
pip install -e ".[testing]"

py.test
