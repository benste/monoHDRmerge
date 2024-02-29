#!/bin/bash
Help()
{
   # Display Help
   echo "Start to UI for interactive HDR merge"
   echo
   echo "Syntax: format [-h]"
   echo "options:"
   echo "h      Print this Help."
   echo "p      set-up python dependencies before launching"
}


setup_poetry=false
while getopts ":hp" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        p) # setup poetry
            setup_poetry=true
    esac
done

# Make sure we exit on any errors
set -e

# Set-up poetry
if [ $setup_poetry = true ]; then
    echo "setting up Poetry"
    sh ./scripts/setup_poetry.sh -d
    echo $'\n\n\n'
fi

# Run UI
echo "Starting UI"
poetry run python monohdrmerge/interface.py