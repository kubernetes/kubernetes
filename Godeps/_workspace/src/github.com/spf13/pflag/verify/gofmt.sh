#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

ROOT=$(dirname "${BASH_SOURCE}")/..

pushd "${ROOT}" > /dev/null

GOFMT=${GOFMT:-"gofmt"}
bad_files=$(find . -name '*.go' | xargs $GOFMT -s -l)
if [[ -n "${bad_files}" ]]; then
  echo "!!! '$GOFMT' needs to be run on the following files: "
  echo "${bad_files}"
  exit 1
fi

# ex: ts=2 sw=2 et filetype=sh
