#!/bin/bash

# Copyright 2018 Google LLC.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# TODO(deklerk) Add integration tests when it's secure to do so. b/64723143

# Fail on any error
set -eo pipefail

# Display commands being run
set -x

# cd to project dir on Kokoro instance
cd git/google_api

go version

# Set $GOPATH
export GOPATH="$HOME/go"
export GOCLOUD_HOME=$GOPATH/src/google.golang.org/api/
export PATH="$GOPATH/bin:$PATH"
export GO111MODULE=on
mkdir -p $GOCLOUD_HOME

# Move code into $GOPATH and get dependencies
git clone . $GOCLOUD_HOME
cd $GOCLOUD_HOME

try3() { eval "$*" || eval "$*" || eval "$*"; }

# All packages, including +build tools, are fetched.
try3 go mod download
./internal/kokoro/vet.sh

# Testing the generator itself depends on a generation step
cd google-api-go-generator; go generate; cd ..

# Run tests and tee output to log file, to be pushed to GCS as artifact.
go test -race -v -short ./... 2>&1 | tee $KOKORO_ARTIFACTS_DIR/$KOKORO_GERRIT_CHANGE_NUMBER.txt
