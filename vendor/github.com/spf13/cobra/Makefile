BIN="./bin"
SRC=$(shell find . -name "*.go")

ifeq (, $(shell which golangci-lint))
$(warning "could not find golangci-lint in $(PATH), run: curl -sfL https://install.goreleaser.com/github.com/golangci/golangci-lint.sh | sh")
endif

.PHONY: fmt lint test install_deps clean

default: all

all: fmt test

fmt:
	$(info ******************** checking formatting ********************)
	@test -z $(shell gofmt -l $(SRC)) || (gofmt -d $(SRC); exit 1)

lint:
	$(info ******************** running lint tools ********************)
	golangci-lint run -v

test: install_deps
	$(info ******************** running tests ********************)
	go test -v ./...

richtest: install_deps
	$(info ******************** running tests with kyoh86/richgo ********************)
	richgo test -v ./...

install_deps:
	$(info ******************** downloading dependencies ********************)
	go get -v ./...

clean:
	rm -rf $(BIN)
