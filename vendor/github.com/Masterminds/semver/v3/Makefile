GOPATH=$(shell go env GOPATH)
GOLANGCI_LINT=$(GOPATH)/bin/golangci-lint

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
fuzz:
	@echo "==> Running Fuzz Tests"
	go env GOCACHE
	go test -fuzz=FuzzNewVersion -fuzztime=15s .
	go test -fuzz=FuzzStrictNewVersion -fuzztime=15s .
	go test -fuzz=FuzzNewConstraint -fuzztime=15s .

$(GOLANGCI_LINT):
	# Install golangci-lint. The configuration for it is in the .golangci.yml
	# file in the root of the repository
	echo ${GOPATH}
	curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh | sh -s -- -b $(GOPATH)/bin v1.56.2
