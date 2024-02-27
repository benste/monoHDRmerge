#!/bin/bash
Help()
{
   # Display Help
   echo "Tests all python files found under monoHDRmerge/"
   echo "using pytest, flake8, black, mypy and isort"
   echo
   echo "Syntax: format [-h]"
   echo "options:"
   echo "h      Print this Help."
   echo
}


integration_test=false
while getopts ":hi" option; do
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

# Run tests
echo "Running Flake8 code linting..."
poetry run flake8 --config scripts/test_config/.flake8 monoHDRmerge/.

echo "Running Black formatting check..."
poetry run black --check --config scripts/test_config/black.toml monoHDRmerge/.

echo "Running isort import sorting check..."
poetry run isort **/*.py -c --settings-file scripts/test_config/.isort.cfg

echo "Running static type-checking with Mypy ..."
poetry run mypy monoHDRmerge/. --config-file scripts/test_config/mypy.ini

echo "Done!"