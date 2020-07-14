.PHONY: default install build test quicktest fmt vet lint 

GO_VERSION := $(shell go version | cut -d' ' -f3 | cut -d. -f2)

# Only use the `-race` flag on newer versions of Go
IS_OLD_GO := $(shell test $(GO_VERSION) -le 2 && echo true)
ifeq ($(IS_OLD_GO),true)
	RACE_FLAG :=
else
	RACE_FLAG := -race -cpu 1,2,4
endif

default: fmt vet lint build quicktest

install:
	go get -t -v ./...

build:
	go build -v ./...

test:
	go test -v $(RACE_FLAG) -cover ./...

quicktest:
	go test ./...

# Capture output and force failure when there is non-empty output
fmt:
	@echo gofmt -l .
	@OUTPUT=`gofmt -l . 2>&1`; \
	if [ "$$OUTPUT" ]; then \
		echo "gofmt must be run on the following files:"; \
		echo "$$OUTPUT"; \
		exit 1; \
	fi

# Only run on go1.5+
vet:
	go tool vet -atomic -bool -copylocks -nilfunc -printf -shadow -rangeloops -unreachable -unsafeptr -unusedresult .

# https://github.com/golang/lint
# go get github.com/golang/lint/golint
# Capture output and force failure when there is non-empty output
# Only run on go1.5+
lint:
	@echo golint ./...
	@OUTPUT=`golint ./... 2>&1`; \
	if [ "$$OUTPUT" ]; then \
		echo "golint errors:"; \
		echo "$$OUTPUT"; \
		exit 1; \
	fi
