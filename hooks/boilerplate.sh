#!/bin/bash

# Print 1 if the file in $1 has the correct boilerplate header, 0 otherwise.
FILE=$1
LINES=$(cat "$(dirname $0)/boilerplate.txt" | wc -l)
DIFFER=$(head -$LINES "${FILE}" | diff -q - "$(dirname $0)/boilerplate.txt")

if [[ -z "${DIFFER}" ]]; then
  echo "${DIFFER}"
  echo "1"
  exit 0
fi

echo "0"
