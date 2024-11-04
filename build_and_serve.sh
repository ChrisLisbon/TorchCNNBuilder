#!/bin/bash

# building doc
echo "-- Building html doc..."

if [[ "$1" == "--force" ]]; then
   rm -rf docs/*.html
fi

pdoc --html -o docs --config latex_math=True torchcnnbuilder/ --force
mv docs/torchcnnbuilder/* docs/
rm -rf docs/torchcnnbuilder

# serving doc
echo "--- Serving docs..."
python -m http.server --directory docs/
