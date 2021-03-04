GO ?= go
CC ?= gcc
ifeq ($(GOPATH),)
export GOPATH := $(shell $(GO) env GOPATH)
endif
FIRST_GOPATH := $(firstword $(subst :, ,$(GOPATH)))
GOBIN := $(shell $(GO) env GOBIN)
ifeq ($(GOBIN),)
	GOBIN := $(FIRST_GOPATH)/bin
endif

all: build test phaul phaul-test

lint:
	@golint -set_exit_status . test phaul
build:
	@$(GO) build -v

test/piggie: test/piggie.c
	@$(CC) $^ -o $@

test/test: test/main.go
	@$(GO) build -v -o test/test test/main.go

test: test/test test/piggie
	mkdir -p image
	test/piggie
	test/test dump `pidof piggie` image
	test/test restore image
	pkill -9 piggie || :

phaul:
	@cd phaul; go build -v

test/phaul: test/phaul-main.go
	@$(GO) build -v -o test/phaul test/phaul-main.go

phaul-test: test/phaul test/piggie
	rm -rf image
	test/piggie
	test/phaul `pidof piggie`
	pkill -9 piggie || :

clean:
	@rm -f test/test test/piggie test/phaul
	@rm -rf image
	@rm -f rpc/rpc.proto

install.tools:
	if [ ! -x "$(GOBIN)/golint" ]; then \
		$(GO) get -u golang.org/x/lint/golint; \
	fi

rpc/rpc.proto:
	curl -s https://raw.githubusercontent.com/checkpoint-restore/criu/master/images/rpc.proto -o $@

rpc/rpc.pb.go: rpc/rpc.proto
	protoc --go_out=. $^

.PHONY: build test clean lint phaul
