#!/bin/bash
# This script is used by the CI to check if 'go generate ./...' is up to date.
#
# Note: If the generated files aren't up to date then this script updates
# them despite printing an error message so running it the second time
# might not print any errors. This isn't very useful locally during development
# but it works well with the CI that downloads a fresh version of the repo
# each time before executing this script.

set -euo pipefail

BASE_DIR="$PWD"
TEMP_DIR=$( mktemp -d )
function cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

cp -r . "${TEMP_DIR}/"
cd $TEMP_DIR
go generate ./...
if ! diff -r . "${BASE_DIR}"; then
    echo
    echo "The generated files aren't up to date."
    echo "Update them with the 'go generate ./...' command."
    exit 1
fi
