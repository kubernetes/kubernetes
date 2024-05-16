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

# Previous bugs have made the tests freeze until the timeout. Golang default
# timeout for tests is 10 minutes, which is too long, considering current tests
# can be executed in less than 1 second. Reduce the timeout, so problems can
# be noticed earlier in the CI.
TEST_TIMEOUT=10s

test:
	go test -v -timeout $(TEST_TIMEOUT)

lint:
	golangci-lint run .
