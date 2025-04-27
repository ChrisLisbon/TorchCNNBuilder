#!/bin/bash

source ./.build/utils.sh

if pip show pdoc | grep "Version: 15.0.0" > /dev/null; then
  print_message $GREEN "-- pdoc==15.0.0 already installed."
else
  print_message $YELLOW "-- Installing pdoc==15.0.0..."
  pip install pdoc==15.0.0
fi

print_message $CYAN "-- Running the documentation generation..."
pdoc --math -d google --no-include-undocumented -t ./.docs/ ./torchcnnbuilder
