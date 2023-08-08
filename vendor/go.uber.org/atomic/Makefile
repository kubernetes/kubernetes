# Directory to place `go install`ed binaries into.
export GOBIN ?= $(shell pwd)/bin

GOLINT = $(GOBIN)/golint
GEN_ATOMICINT = $(GOBIN)/gen-atomicint
GEN_ATOMICWRAPPER = $(GOBIN)/gen-atomicwrapper
STATICCHECK = $(GOBIN)/staticcheck

GO_FILES ?= $(shell find . '(' -path .git -o -path vendor ')' -prune -o -name '*.go' -print)

# Also update ignore section in .codecov.yml.
COVER_IGNORE_PKGS = \
	go.uber.org/atomic/internal/gen-atomicint \
	go.uber.org/atomic/internal/gen-atomicwrapper

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
	cd tools && go install golang.org/x/lint/golint

$(STATICCHECK):
	cd tools && go install honnef.co/go/tools/cmd/staticcheck

$(GEN_ATOMICWRAPPER): $(wildcard ./internal/gen-atomicwrapper/*)
	go build -o $@ ./internal/gen-atomicwrapper

$(GEN_ATOMICINT): $(wildcard ./internal/gen-atomicint/*)
	go build -o $@ ./internal/gen-atomicint

.PHONY: golint
golint: $(GOLINT)
	$(GOLINT) ./...

.PHONY: staticcheck
staticcheck: $(STATICCHECK)
	$(STATICCHECK) ./...

.PHONY: lint
lint: gofmt golint staticcheck generatenodirty

# comma separated list of packages to consider for code coverage.
COVER_PKG = $(shell \
	go list -find ./... | \
	grep -v $(foreach pkg,$(COVER_IGNORE_PKGS),-e "^$(pkg)$$") | \
	paste -sd, -)

.PHONY: cover
cover:
	go test -coverprofile=cover.out -coverpkg  $(COVER_PKG) -v ./...
	go tool cover -html=cover.out -o cover.html

.PHONY: generate
generate: $(GEN_ATOMICINT) $(GEN_ATOMICWRAPPER)
	go generate ./...

.PHONY: generatenodirty
generatenodirty:
	@[ -z "$$(git status --porcelain)" ] || ( \
		echo "Working tree is dirty. Commit your changes first."; \
		git status; \
		exit 1 )
	@make generate
	@status=$$(git status --porcelain); \
		[ -z "$$status" ] || ( \
		echo "Working tree is dirty after `make generate`:"; \
		echo "$$status"; \
		echo "Please ensure that the generated code is up-to-date." )
