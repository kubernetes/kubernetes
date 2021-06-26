GO ?= go
CC ?= gcc

all: build test phaul-test

lint:
	golangci-lint run ./...

build:
	$(GO) build -v ./...

TEST_BINARIES := test/test test/piggie/piggie test/phaul/phaul
test-bin: $(TEST_BINARIES)

test/piggie/piggie: test/piggie/piggie.c
	$(CC) $^ -o $@

test/test: test/*.go
	$(GO) build -v -o $@ $^

test: $(TEST_BINARIES)
	mkdir -p image
	PID=$$(test/piggie/piggie) && { \
	test/test dump $$PID image && \
	test/test restore image; \
	pkill -9 piggie; \
	}
	rm -rf image

test/phaul/phaul: test/phaul/*.go
	$(GO) build -v -o $@ $^

phaul-test: $(TEST_BINARIES)
	rm -rf image
	PID=$$(test/piggie/piggie) && { \
	test/phaul/phaul $$PID; \
	pkill -9 piggie; \
	}

clean:
	@rm -f $(TEST_BINARIES)
	@rm -rf image
	@rm -f rpc/rpc.proto stats/stats.proto

rpc/rpc.proto:
	curl -sSL https://raw.githubusercontent.com/checkpoint-restore/criu/master/images/rpc.proto -o $@

stats/stats.proto:
	curl -sSL https://raw.githubusercontent.com/checkpoint-restore/criu/master/images/stats.proto -o $@

rpc/rpc.pb.go: rpc/rpc.proto
	protoc --go_out=. $^

stats/stats.pb.go: stats/stats.proto
	protoc --go_out=. $^

.PHONY: build test phaul-test test-bin clean lint
