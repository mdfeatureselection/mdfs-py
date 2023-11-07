#!/bin/sh

set -e

rm -rf dist/
python3 -m build
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install ./dist/mdfs-*.*.*-py3-none-*.whl
