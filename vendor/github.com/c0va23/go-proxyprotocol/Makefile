GOARCH := $(shell go env GOARCH)
GOOS := $(shell go env GOOS)

GOLANGCI_LINT_VERSION := 1.16.0
GOLANGCI_LINT_ARCHIVE_NAME := golangci-lint-${GOLANGCI_LINT_VERSION}-${GOOS}-${GOARCH}
GOLANGCI_LINT_URL := https://github.com/golangci/golangci-lint/releases/download/v${GOLANGCI_LINT_VERSION}/${GOLANGCI_LINT_ARCHIVE_NAME}.tar.gz

export PATH := $(PWD)/bin:$(PATH)
export GO111MODULE=on

default: lint test

bin/${GOLANGCI_LINT_ARCHIVE_NAME}/:
	mkdir -p bin
	curl -L ${GOLANGCI_LINT_URL} | tar --directory bin/ --gzip --extract --verbose

bin/golangci-lint: bin/${GOLANGCI_LINT_ARCHIVE_NAME}/
	ln -f -s $(PWD)/bin/${GOLANGCI_LINT_ARCHIVE_NAME}/golangci-lint $@
	touch $@

lint: bin/golangci-lint
	golangci-lint run ./...

test:
	go test ./...