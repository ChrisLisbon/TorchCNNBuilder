source ./.build/utils.sh

echo "-- Checking for required linter packages..."

check_and_install() {
    PACKAGE=$1
    pip show $PACKAGE > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        print_message $YELLOW "-- $PACKAGE is not installed. Installing..."
        pip install $PACKAGE
    else
        print_message $GREEN "-- $PACKAGE is already installed."
    fi
}

# checking and installing packages
check_and_install flake8
check_and_install black
check_and_install isort

# checking flake8 config
FLAKE8_CONFIG=".github/workflows/.flake8"
if [ ! -f "$FLAKE8_CONFIG" ]; then
    print_message $YELLOW "-- Config file $FLAKE8_CONFIG is not found. Creating a new one."
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

print_message $CYAN "-- Code sorting with isort..."
isort --profile black $CHECK_PATH

print_message $CYAN "-- Code formatting with black..."
black --line-length 120 $CHECK_PATH

print_message $CYAN "-- Code checking with flake8..."
flake8 --config $FLAKE8_CONFIG $CHECK_PATH

print_message $GREEN "-- Done."