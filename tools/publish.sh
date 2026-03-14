#!/bin/sh
# Publish SC-BIG as sc-big-caller to PyPI. Requires a valid
# username and API token in your .pypirc.
python3 -m pip install --upgrade build
python3 -m build
python3 -m twine upload --verbose dist/*
