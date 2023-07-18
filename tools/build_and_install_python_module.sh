#!/bin/sh

set -e

rm -rf dist/
python -m build
python -m pip uninstall --yes mdfs
python -m pip install ./dist/mdfs-*.*.*-py3-none-*.whl
