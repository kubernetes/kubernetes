#!/bin/bash
set -e

# Build a Go static web server on top of busybox image
# and compile it for target daemon

dir="$DEST/httpserver"
mkdir -p "$dir"
(
	cd "$dir"
	GOOS=linux GOARCH=amd64 go build -o httpserver github.com/docker/docker/contrib/httpserver
	cp ../../../../contrib/httpserver/Dockerfile .
	docker build -qt httpserver . > /dev/null
)
rm -rf "$dir"
