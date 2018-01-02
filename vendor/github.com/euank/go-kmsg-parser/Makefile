.PHONY: all godmesg test test-deps

all: godmesg

godmesg:
	go build -o ./bin/godmesg ./cmd/godmesg

test:
	go test -v ./...

test-deps:
	go get github.com/stretchr/testify/assert
