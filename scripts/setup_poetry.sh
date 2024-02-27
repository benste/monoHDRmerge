Help()
{
   # Display Help
   echo "Installs pip 23.3.1 and poetry 1.7.1 followed by"
   echo "installation of dependencies through poetry"
   echo
   echo "Syntax: setup_poetry [-d|-h]"
   echo "options:"
   echo "d      Also installs dev dependencies."
   echo "h      Print this Help."
   echo
}

Setup()
{
    #!/bin/bash
    pip install "pip==23.3.1"
    pip install "poetry==1.7.1"
    poetry config experimental.new-installer false

    # Project initialization:
    poetry config virtualenvs.create true
    if [ $install_dev = true ]; then
        poetry install --no-interaction --no-ansi --no-root
    else
        poetry install --no-interaction --no-ansi --no-root --no-dev
    fi
}


install_dev=false
while getopts ":hd" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        d) # Install dev dependencies?
            install_dev=true;;
    esac
done

Setup


