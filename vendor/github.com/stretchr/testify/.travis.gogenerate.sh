#!/bin/bash

if [[ "$TRAVIS_GO_VERSION" =~ ^1\.[45](\..*)?$ ]]; then
  exit 0
fi

go get github.com/ernesto-jimenez/gogen/imports
go generate ./...
if [ -n "$(git diff)" ]; then
  echo "Go generate had not been run"
  git diff
  exit 1
fi
