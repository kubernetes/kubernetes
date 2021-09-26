#!/bin/bash

# If GOMOD is defined we are running with Go Modules enabled, either
# automatically or via the GO111MODULE=on environment variable. Codegen only
# works with modules, so skip generation if modules is not in use.
if [[ -z "$(go env GOMOD)" ]]; then
  echo "Skipping go generate because modules not enabled and required"
  exit 0
fi

go generate ./...
if [ -n "$(git diff)" ]; then
  echo "Go generate had not been run"
  git diff
  exit 1
fi
