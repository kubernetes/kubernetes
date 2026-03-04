# Directory containing the Makefile.
PROJECT_ROOT = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

export GOBIN = $(PROJECT_ROOT)/bin
export PATH := $(GOBIN):$(PATH)

GO_FILES = $(shell find . \
	   -path '*/.*' -prune -o \
	   '(' -type f -a -name '*.go' ')' -print)

# Additional test flags.
TEST_FLAGS ?=

.PHONY: all
all: lint build test

.PHONY: lint
lint: golangci-lint tidy-lint

.PHONY: build
build:
	go build ./...

.PHONY: test
test:
	go test -v -race ./...
	go test -v -trace=/dev/null .

.PHONY: cover
cover:
	go test -race -coverprofile=cover.out -coverpkg=./... ./...
	go tool cover -html=cover.out -o cover.html

.PHONY: golangci-lint
golangci-lint:
	golangci-lint run

.PHONY: tidy
tidy:
	go mod tidy

.PHONY: tidy-lint
tidy-lint:
	go mod tidy
	git diff --exit-code -- go.mod go.sum
