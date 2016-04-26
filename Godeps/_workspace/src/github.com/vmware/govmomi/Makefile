.PHONY: test

all: check test

check: goimports govet

goimports:
	@echo checking go imports...
	@! goimports -d . 2>&1 | egrep -v '^$$'

govet:
	@echo checking go vet...
	@go tool vet -structtags=false -methods=false .

test:
	go get
	go test -v $(TEST_OPTS) ./...

install:
	go install github.com/vmware/govmomi/govc
