PROTO_PATH := ${GOPATH}/src:.
PROTO_PATH := ${PROTO_PATH}:${GOPATH}/src/github.com/gogo/protobuf/protobuf
PROTO_PATH := ${PROTO_PATH}:${GOPATH}/src/github.com/gogo/protobuf/gogoproto

.PHONY: all

all:
	protoc --proto_path=${PROTO_PATH} --gogo_out=. *.proto
	protoc --proto_path=${PROTO_PATH} --gogo_out=. ./scheduler/*.proto
