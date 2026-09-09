# Project packages.
PACKAGES=$(shell go list ./...)

# Flags passed to `go test`
BUILDFLAGS ?= 
TESTFLAGS ?= 

.PHONY: all build test coverage
.DEFAULT: all

all: build

build: ## no binaries to build, so just check compilation suceeds
	go build ${BUILDFLAGS} ./...

test: ## run tests
	go test ${TESTFLAGS} ./...

coverage: ## generate coverprofiles from the unit tests
	rm -f coverage.txt
	go test ${TESTFLAGS} -cover -coverprofile=cover.out ./...

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_\/%-]+:.*?##/ { printf "  \033[36m%-27s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
