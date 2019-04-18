#!/usr/bin/env bash

# Setup python environment if it does not exist
echo "Setting up environment for project..."
if [ -d "./venv" ]; then
  echo "./venv directory already exist"

  source ./venv/bin/activate
else
  python3 -m venv venv

  source ./venv/bin/activate
fi

if [ $? -ne 0 ]; then
  echo "Error in configuring virtual environment"
  exit 1
fi

# Virtual Environment activated
# Upgrade pip to latest version
pip3 install -U pip

# Installing all pre-requisite packages
cd tmp2
pip3 install -e .

cd ..

# Install NLTK Data
mkdir ./venv/nltk_data
python3 -m nltk.downloader -d "./venv/nltk_data" all

echo "Setup completed."

