#!/bin/bash
# This script is used by the CI to check if the code is gofmt formatted.

set -euo pipefail

GOFMT_DIFF=$(IFS=$'\n'; gofmt -d $( find . -type f -name '*.go' ) )
if [[ -n "${GOFMT_DIFF}" ]]; then
    echo "${GOFMT_DIFF}"
    echo
    echo "The go source files aren't gofmt formatted."
    exit 1
fi
