TEST?=./...

default: test

fmt: generate
	go fmt ./...

test: generate
	go get -t ./...
	go test $(TEST) $(TESTARGS)

generate:
	go generate ./...

updatedeps:
	go get -u golang.org/x/tools/cmd/stringer

.PHONY: default generate test updatedeps
