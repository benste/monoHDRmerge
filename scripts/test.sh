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
   echo "p      setup dependencies through poetry before running the tests"
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

# Run tests
echo "Running Flake8 code linting..."
poetry run flake8 --config scripts/test_config/.flake8 .

echo "Running Black formatting check..."
poetry run black --check --config scripts/test_config/black.toml .

echo "Running isort import sorting check..."
poetry run isort **/*.py -c --settings-file scripts/test_config/.isort.cfg

echo "Running static type-checking with Mypy ..."
poetry run mypy . --config-file scripts/test_config/mypy.ini

echo "Done!"