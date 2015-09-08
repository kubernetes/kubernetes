#!/bin/bash

ROOT=$(dirname "${BASH_SOURCE}")/..
GOLINT=${GOLINT:-"golint"}

pushd "${ROOT}" > /dev/null
  bad_files=$($GOLINT -min_confidence=0.9 ./...)
  if [[ -n "${bad_files}" ]]; then
    echo "!!! '$GOLINT' problems: "
    echo "${bad_files}"
    exit 1
  fi
popd > /dev/null

# ex: ts=2 sw=2 et filetype=sh
