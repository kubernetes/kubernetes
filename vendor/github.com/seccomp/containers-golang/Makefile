export GO111MODULE=off

TAGS ?= seccomp
BUILDFLAGS := -tags "$(AUTOTAGS) $(TAGS)"
GO := go
PACKAGE := github.com/seccomp/containers-golang

sources := $(wildcard *.go)

.PHONY: seccomp.json
seccomp.json: $(sources)
	$(GO) build -compiler gc $(BUILDFLAGS) ./cmd/generate.go
	$(GO) build -compiler gc ./cmd/generate.go
	$(GO) run ${BUILDFLAGS} cmd/generate.go

all: seccomp.json

.PHONY: test-unit
test-unit:
	$(GO) test -v $(BUILDFLAGS) $(shell $(GO) list ./... | grep -v ^$(PACKAGE)/vendor)
	$(GO) test -v $(shell $(GO) list ./... | grep -v ^$(PACKAGE)/vendor)

.PHONY: vendor
vendor:
	export GO111MODULE=on \
		$(GO) mod tidy && \
		$(GO) mod vendor && \
		$(GO) mod verify

.PHONY: clean
clean:
	rm -f generate
