GOPATH=$(shell go env GOPATH)
GOLANGCI_LINT=$(GOPATH)/bin/golangci-lint
GOFUZZBUILD = $(GOPATH)/bin/go-fuzz-build
GOFUZZ = $(GOPATH)/bin/go-fuzz

.PHONY: lint
lint: $(GOLANGCI_LINT)
	@echo "==> Linting codebase"
	@$(GOLANGCI_LINT) run

.PHONY: test
test:
	@echo "==> Running tests"
	GO111MODULE=on go test -v

.PHONY: test-cover
test-cover:
	@echo "==> Running Tests with coverage"
	GO111MODULE=on go test -cover .

.PHONY: fuzz
fuzz: $(GOFUZZBUILD) $(GOFUZZ)
	@echo "==> Fuzz testing"
	$(GOFUZZBUILD)
	$(GOFUZZ) -workdir=_fuzz

$(GOLANGCI_LINT):
	# Install golangci-lint. The configuration for it is in the .golangci.yml
	# file in the root of the repository
	echo ${GOPATH}
	curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh | sh -s -- -b $(GOPATH)/bin v1.17.1

$(GOFUZZBUILD):
	cd / && go get -u github.com/dvyukov/go-fuzz/go-fuzz-build

$(GOFUZZ):
	cd / && go get -u github.com/dvyukov/go-fuzz/go-fuzz github.com/dvyukov/go-fuzz/go-fuzz-dep