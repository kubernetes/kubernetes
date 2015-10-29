#!/usr/bin/env bash
set -e

# This script runs all validations

validate() {
    export MAKEDIR=/go/src/github.com/docker/docker/hack/make
    sed -i 's!docker/docker!docker/libcontainer!' /go/src/github.com/docker/docker/hack/make/.validate
    bash /go/src/github.com/docker/docker/hack/make/validate-dco
    bash /go/src/github.com/docker/docker/hack/make/validate-gofmt
    go get golang.org/x/tools/cmd/vet
    go vet github.com/docker/libcontainer/...
}

# run validations
validate
