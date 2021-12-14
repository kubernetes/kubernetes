SHELL = /bin/bash
GO ?= go
CC ?= gcc
COVERAGE_PATH ?= $(shell pwd)/.coverage

all: build test phaul-test

lint:
	golangci-lint run ./...

build:
	$(GO) build -v ./...

TEST_PAYLOAD := test/piggie/piggie
TEST_BINARIES := test/test $(TEST_PAYLOAD) test/phaul/phaul
COVERAGE_BINARIES := test/test.coverage test/phaul/phaul.coverage
test-bin: $(TEST_BINARIES)

test/piggie/piggie: test/piggie/piggie.c
	$(CC) $^ -o $@

test/test: test/main.go
	$(GO) build -v -o $@ $^

test: $(TEST_BINARIES)
	mkdir -p image
	PID=$$(test/piggie/piggie) && { \
	test/test dump $$PID image && \
	test/test restore image; \
	pkill -9 piggie; \
	}
	rm -rf image

test/phaul/phaul: test/phaul/main.go
	$(GO) build -v -o $@ $^

phaul-test: $(TEST_BINARIES)
	rm -rf image
	PID=$$(test/piggie/piggie) && { \
	test/phaul/phaul $$PID; \
	pkill -9 piggie; \
	}

test/test.coverage: test/*.go
	$(GO) test \
		-covermode=count \
		-coverpkg=./... \
		-mod=vendor \
		-tags coverage \
		-buildmode=pie -c -o $@ $^

test/phaul/phaul.coverage: test/phaul/*.go
	$(GO) test \
		-covermode=count \
		-coverpkg=./... \
		-mod=vendor \
		-tags coverage \
		-buildmode=pie -c -o $@ $^

coverage: $(COVERAGE_BINARIES) $(TEST_PAYLOAD)
	mkdir -p $(COVERAGE_PATH)
	mkdir -p image
	PID=$$(test/piggie/piggie) && { \
	test/test.coverage -test.coverprofile=coverprofile.integration.$$RANDOM -test.outputdir=${COVERAGE_PATH} COVERAGE dump $$PID image && \
	test/test.coverage -test.coverprofile=coverprofile.integration.$$RANDOM -test.outputdir=${COVERAGE_PATH} COVERAGE restore image; \
	pkill -9 piggie; \
	}
	rm -rf image
	PID=$$(test/piggie/piggie) && { \
	test/phaul/phaul.coverage -test.coverprofile=coverprofile.integration.$$RANDOM -test.outputdir=${COVERAGE_PATH} COVERAGE $$PID; \
	pkill -9 piggie; \
	}

clean:
	@rm -f $(TEST_BINARIES) $(COVERAGE_BINARIES) codecov
	@rm -rf image $(COVERAGE_PATH)

rpc/rpc.proto:
	curl -sSL https://raw.githubusercontent.com/checkpoint-restore/criu/master/images/rpc.proto -o $@

stats/stats.proto:
	curl -sSL https://raw.githubusercontent.com/checkpoint-restore/criu/master/images/stats.proto -o $@

rpc/rpc.pb.go: rpc/rpc.proto
	protoc --go_out=. --go_opt=M$^=rpc/ $^

stats/stats.pb.go: stats/stats.proto
	protoc --go_out=. --go_opt=M$^=stats/ $^

vendor:
	GO111MODULE=on $(GO) mod tidy
	GO111MODULE=on $(GO) mod vendor
	GO111MODULE=on $(GO) mod verify

codecov:
	curl -Os https://uploader.codecov.io/latest/linux/codecov
	chmod +x codecov
	./codecov -f '.coverage/*'

.PHONY: build test phaul-test test-bin clean lint vendor coverage codecov
