#!/bin/bash
set -e
mypy --disallow-untyped-defs --explicit-package-bases .
ruff check --target-version=py39 --fix .
ruff format
