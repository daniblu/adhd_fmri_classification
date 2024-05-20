#!/bin/bash

# Get the directory of the Bash script
scriptDir=$(dirname -- "$(readlink -f -- "$BASH_SOURCE")")

# Activate the virtual environment
source "$scriptDir/env/bin/activate"

# Install kernel for notebooks
python -m ipykernel install --user --name=env

echo "Done!"