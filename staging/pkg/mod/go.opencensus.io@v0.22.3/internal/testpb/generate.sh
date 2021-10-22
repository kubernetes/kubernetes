#!/bin/sh
# generate .pb.go file from .proto file.
set -e
protoc --go_out=plugins=grpc:. test.proto
echo '//go:generate ./generate.sh
' >> test.pb.go

