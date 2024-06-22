#!/bin/bash

echo "-- Downloading linter packages..."
pip install flake8 black isort

# checking flake8 config
FLAKE8_CONFIG=".github/workflows/.flake8"
if [ ! -f "$FLAKE8_CONFIG" ]; then
    echo "-- Config file $FLAKE8_CONFIG is not found. Creating a new one."
    cat <<EOL > $FLAKE8_CONFIG
[flake8]
max-line-length = 120
extend-ignore = E203, E266, W503, W504
show-source = true
statistics = true
plugins = flake8-isort, flake8-black
ignore = D1
EOL
fi

# running linter
CHECK_PATH="torchcnnbuilder"

echo "-- Code sorting with isort..."
isort --profile black $CHECK_PATH

echo "-- Code formatting with black..."
black --line-length 120 $CHECK_PATH

echo "-- Code checking with flake8..."
flake8 --config $FLAKE8_CONFIG $CHECK_PATH

echo "-- Done."
