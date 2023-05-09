# libseccomp-golang

.PHONY: all check check-build check-syntax fix-syntax vet test lint

all: check-build

check: lint test

check-build:
	go build

check-syntax:
	gofmt -d .

fix-syntax:
	gofmt -w .

vet:
	go vet -v ./...

test:
	go test -v

lint:
	golangci-lint run .
