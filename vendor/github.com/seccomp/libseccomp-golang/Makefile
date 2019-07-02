# libseccomp-golang

.PHONY: all check check-build check-syntax fix-syntax vet test lint

all: check-build

check: vet test

check-build:
	go build

check-syntax:
	gofmt -d .

fix-syntax:
	gofmt -w .

vet:
	go vet -v

test:
	go test -v

lint:
	@$(if $(shell which golint),true,$(error "install golint and include it in your PATH"))
	golint -set_exit_status
