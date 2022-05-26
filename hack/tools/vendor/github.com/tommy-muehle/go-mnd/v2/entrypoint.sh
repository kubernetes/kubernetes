#!/usr/bin/env bash

# Expand the arguments into an array of strings. This is required because the GitHub action
# provides all arguments concatenated as a single string.
ARGS=("$@")

/bin/go-mnd "${ARGS[*]}"
