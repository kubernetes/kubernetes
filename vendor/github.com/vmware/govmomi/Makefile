.PHONY: test

all: check test

check: goimports govet

goimports:
	@echo checking go imports...
	@go get golang.org/x/tools/cmd/goimports
	@! goimports -d . 2>&1 | egrep -v '^$$'

govet:
	@echo checking go vet...
	@go tool vet -structtags=false -methods=false $$(find . -mindepth 1 -maxdepth 1 -type d -not -name vendor)

install:
	go install -v github.com/vmware/govmomi/govc
	go install -v github.com/vmware/govmomi/vcsim

go-test:
	go test -v $(TEST_OPTS) ./...

govc-test: install
	(cd govc/test && ./vendor/github.com/sstephenson/bats/libexec/bats -t .)

test: go-test govc-test

doc: install
	./govc/usage.sh > ./govc/USAGE.md
