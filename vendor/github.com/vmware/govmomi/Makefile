.PHONY: test

all: check test

check: goimports govet

vendor:
	go get golang.org/x/tools/cmd/goimports
	go get github.com/davecgh/go-spew/spew
	go get golang.org/x/net/context

goimports: vendor
	@echo checking go imports...
	@! goimports -d . 2>&1 | egrep -v '^$$'

govet:
	@echo checking go vet...
	@go tool vet -structtags=false -methods=false .

test: vendor
	go test -v $(TEST_OPTS) ./...

install: vendor
	go install github.com/vmware/govmomi/govc
