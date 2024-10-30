# building doc
pdoc --html -o docs --config latex_math=True torchcnnbuilder/ --force

# serving doc
python -m http.server --directory docs/
