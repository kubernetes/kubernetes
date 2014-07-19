#!/bin/bash

set -e
set -x

go build --ldflags '-extldflags "-static" -s' pause.go
