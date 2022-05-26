.PHONY: clean check test build

default: clean check test build

clean:
	rm -rf dist/ cover.out

test: clean
	go test -v -cover ./...

check:
	golangci-lint run

build:
	go build -ldflags "-s -w" -trimpath ./cmd/tagliatelle/
