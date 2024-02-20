#!/bin/bash

# Get a list of all conda environments
environments=$(conda env list | awk '{print $1}' | tail -n +3)

for env in $environments
do
    echo "Checking environment $env"
    
    # Activate the environment
    conda activate $env

    # Check if transformers is installed
    if pip list | grep -q transformers; then
        echo "Transformers is installed in $env"
    else
        echo "Transformers is not installed in $env"
    fi
done