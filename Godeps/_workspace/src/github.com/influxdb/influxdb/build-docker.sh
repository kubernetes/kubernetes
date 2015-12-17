#!/bin/sh

set -e -x

GO_VER=${GO_VER:-1.5}

docker run -it -v "${GOPATH}":/gopath -v "$(pwd)":/app -e "GOPATH=/gopath" -w /app golang:$GO_VER sh -c 'CGO_ENABLED=0 go build -a --installsuffix cgo --ldflags="-s" -o influxd ./cmd/influxd'

docker build -t influxdb .
