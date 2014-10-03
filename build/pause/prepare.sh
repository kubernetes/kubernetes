#!/bin/bash

set -e
set -x

# Build the binary.
go build --ldflags '-extldflags "-static" -s' pause.go

# Run goupx to shrink binary size.
go get github.com/pwaller/goupx
goupx pause
