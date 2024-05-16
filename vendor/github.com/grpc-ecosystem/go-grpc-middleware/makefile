SHELL=/bin/bash

GOFILES_NOVENDOR = $(shell go list ./... | grep -v /vendor/)

all: vet fmt test

fmt:
	go fmt $(GOFILES_NOVENDOR)

vet:
	# do not check lostcancel, they are intentional.
	go vet -lostcancel=false $(GOFILES_NOVENDOR)

test: vet
	./scripts/test_all.sh

.PHONY: all test
