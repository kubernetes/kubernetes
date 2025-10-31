#!/bin/sh

cd $(dirname $0)
exec go run conformance.go $*