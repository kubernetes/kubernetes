#!/bin/bash

set -e -x

export GOPATH="$JENKINS_HOME/workspace/project"
export GOBIN="$GOPATH/bin"
export PATH="$GOBIN:$PATH"

# Kubernetes version(s) to run the integration tests against.
kube_version="1.2.4"

if ! git diff --name-only origin/master | grep -c -E "*.go|*.sh|.*yaml|Makefile" &> /dev/null; then
  echo "This PR does not touch files that require integration testing. Skipping integration tests!"
  exit 0
fi

make -e SUPPORTED_KUBE_VERSIONS=$kube_version test-unit test-integration
