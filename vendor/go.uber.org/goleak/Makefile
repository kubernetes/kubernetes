export GOBIN ?= $(shell pwd)/bin

REVIVE = $(GOBIN)/revive

GO_FILES := $(shell \
	find . '(' -path '*/.*' -o -path './vendor' ')' -prune \
	-o -name '*.go' -print | cut -b3-)

.PHONY: build
build:
	go build ./...

.PHONY: install
install:
	go mod download

.PHONY: test
test:
	go test -v -race ./...
	go test -v -trace=/dev/null .

.PHONY: cover
cover:
	go test -race -coverprofile=cover.out -coverpkg=./... ./...
	go tool cover -html=cover.out -o cover.html

$(REVIVE):
	cd tools && go install github.com/mgechev/revive

.PHONY: lint
lint: $(REVIVE)
	@rm -rf lint.log
	@echo "Checking formatting..."
	@gofmt -d -s $(GO_FILES) 2>&1 | tee lint.log
	@echo "Checking vet..."
	@go vet ./... 2>&1 | tee -a lint.log
	@echo "Checking lint..."
	@$(REVIVE) -set_exit_status ./... 2>&1 | tee -a lint.log
	@echo "Checking for unresolved FIXMEs..."
	@git grep -i fixme | grep -v -e '^vendor/' -e '^Makefile' | tee -a lint.log
	@[ ! -s lint.log ]
