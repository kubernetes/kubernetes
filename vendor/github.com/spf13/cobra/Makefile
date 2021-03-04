BIN="./bin"
SRC=$(shell find . -name "*.go")

ifeq (, $(shell which richgo))
$(warning "could not find richgo in $(PATH), run: go get github.com/kyoh86/richgo")
endif

.PHONY: fmt vet test cobra_generator install_deps clean

default: all

all: fmt vet test cobra_generator	

fmt:
	$(info ******************** checking formatting ********************)
	@test -z $(shell gofmt -l $(SRC)) || (gofmt -d $(SRC); exit 1)

test: install_deps vet
	$(info ******************** running tests ********************)
	richgo test -v ./...

cobra_generator: install_deps
	$(info ******************** building generator ********************)
	mkdir -p $(BIN)
	make -C cobra all

install_deps:
	$(info ******************** downloading dependencies ********************)
	go get -v ./...

vet:
	$(info ******************** vetting ********************)
	go vet ./...

clean:
	rm -rf $(BIN)
