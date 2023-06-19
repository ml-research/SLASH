#!/bin/bash

source ~/.bashrc
pyenv uninstall venv_slash
pyenv virtualenv venv_slash
pyenv activate venv_slash
pip install --upgrade pip
pip install -r requirements.txt
