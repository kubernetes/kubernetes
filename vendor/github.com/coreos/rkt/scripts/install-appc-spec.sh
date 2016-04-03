#!/bin/bash

set -e

GOPATH=${GOPATH:-"/go"}
mkdir -p $GOPATH
go get github.com/appc/spec/...
