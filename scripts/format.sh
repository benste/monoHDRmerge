#!/bin/bash
Help()
{
   # Display Help
   echo "Formats all python files found using black and isort"
   echo
   echo "Syntax: format [-h]"
   echo "options:"
   echo "h      Print this Help."
   echo
}


while getopts ":h" option; do
    case $option in
        h) # display Help
            Help
            exit;;
    esac
done


# Set-up poetry
sh ./script/setup_poetry.sh -d

# Make sure we exit on any errors
set -e

# Apply formatting
echo "Running Black formatting..."
poetry run black --config scripts/test_config/black.toml .
echo "Running isort..."
poetry run isort **/*.py --settings-file scripts/test_config/.isort.cfg
echo "Done!"