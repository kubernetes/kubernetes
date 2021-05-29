# Directory to place `go install`ed binaries into.
export GOBIN ?= $(shell pwd)/bin

GOLINT = $(GOBIN)/golint

GO_FILES ?= *.go

.PHONY: build
build:
	go build ./...

.PHONY: test
test:
	go test -race ./...

.PHONY: gofmt
gofmt:
	$(eval FMT_LOG := $(shell mktemp -t gofmt.XXXXX))
	gofmt -e -s -l $(GO_FILES) > $(FMT_LOG) || true
	@[ ! -s "$(FMT_LOG)" ] || (echo "gofmt failed:" && cat $(FMT_LOG) && false)

$(GOLINT):
	go install golang.org/x/lint/golint

.PHONY: golint
golint: $(GOLINT)
	$(GOLINT) ./...

.PHONY: lint
lint: gofmt golint

.PHONY: cover
cover:
	go test -coverprofile=cover.out -coverpkg ./... -v ./...
	go tool cover -html=cover.out -o cover.html
