#!/bin/bash

# building doc
echo "-- Building html doc..."

if [[ "$1" == "--force" ]]; then
    rm -rf docs/torchcnnbuilder/
fi

pdoc --html -o docs --config latex_math=True torchcnnbuilder/ --force

# serving doc
echo "--- Serving docs..."
python -m http.server --directory docs/
