# Directory to put `go install`ed binaries in.
export GOBIN ?= $(shell pwd)/bin

GO_FILES := $(shell \
	find . '(' -path '*/.*' -o -path './vendor' ')' -prune \
	-o -name '*.go' -print | cut -b3-)

.PHONY: build
build:
	go build ./...

.PHONY: test
test:
	go test -race ./...

.PHONY: gofmt
gofmt:
	$(eval FMT_LOG := $(shell mktemp -t gofmt.XXXXX))
	@gofmt -e -s -l $(GO_FILES) > $(FMT_LOG) || true
	@[ ! -s "$(FMT_LOG)" ] || (echo "gofmt failed:" | cat - $(FMT_LOG) && false)

.PHONY: golint
golint:
	@cd tools && go install golang.org/x/lint/golint
	@$(GOBIN)/golint ./...

.PHONY: staticcheck
staticcheck:
	@cd tools && go install honnef.co/go/tools/cmd/staticcheck
	@$(GOBIN)/staticcheck ./...

.PHONY: lint
lint: gofmt golint staticcheck

.PHONY: cover
cover:
	go test -coverprofile=cover.out -coverpkg=./... -v ./...
	go tool cover -html=cover.out -o cover.html

update-license:
	@cd tools && go install go.uber.org/tools/update-license
	@$(GOBIN)/update-license $(GO_FILES)
