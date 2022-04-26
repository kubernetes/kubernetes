.PHONY: all imports test lint

all: imports test lint

imports:
	goimports -w ./

test:
	go test -race ./...

test_coverage:
	go test -race -coverprofile=coverage.out -covermode=atomic ./...

lint:
	golangci-lint run ./...

