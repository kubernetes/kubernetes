#!/bin/bash

set -e
set -x

CGO_ENABLED=0 go build  -a -ldflags '-extldflags "-static" -s' hello.go
