#!/bin/bash

set -e -x

export GOPATH="$JENKINS_HOME/workspace/project"
export GOBIN="$GOPATH/bin"
export PATH="$GOBIN:$PATH"

if ! git diff --name-only origin/master | grep -c -E "*.go|*.sh|.*yaml|Makefile" &> /dev/null; then
  echo "This PR does not touch files that require integration testing. Skipping integration tests!"
  exit 0
fi

make test-unit test-integration
