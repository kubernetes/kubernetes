#!/bin/bash
# This script is used by CI to check if the code passes golint.

set -u

if ! command -v golint >/dev/null; then
    echo "error: golint not found; go get -u golang.org/x/lint/golint" >&2
    exit 1
fi

GOLINT_OUTPUT=$(IFS=$'\n'; golint ./... | grep -v "mockgen/internal/.*\|sample/.*")
if [[ -n "${GOLINT_OUTPUT}" ]]; then
    echo "${GOLINT_OUTPUT}"
    echo
    echo "The go source files aren't passing golint."
    exit 1
fi
