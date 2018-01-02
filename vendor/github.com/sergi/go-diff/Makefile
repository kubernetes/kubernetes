.PHONY: all clean clean-coverage install install-dependencies install-tools lint test test-verbose test-with-coverage

export ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
export PKG := github.com/sergi/go-diff
export ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

$(eval $(ARGS):;@:) # turn arguments into do-nothing targets
export ARGS

ifdef ARGS
	PKG_TEST := $(ARGS)
else
	PKG_TEST := $(PKG)/...
endif

all: install-tools install-dependencies install lint test

clean:
	go clean -i $(PKG)/...
	go clean -i -race $(PKG)/...
clean-coverage:
	find $(ROOT_DIR) | grep .coverprofile | xargs rm
install:
	go install -v $(PKG)/...
install-dependencies:
	go get -t -v $(PKG)/...
	go build -v $(PKG)/...
install-tools:
	# Install linting tools
	go get -u -v github.com/golang/lint/...
	go get -u -v github.com/kisielk/errcheck/...

	# Install code coverage tools
	go get -u -v github.com/onsi/ginkgo/ginkgo/...
	go get -u -v github.com/modocache/gover/...
	go get -u -v github.com/mattn/goveralls/...
lint:
	$(ROOT_DIR)/scripts/lint.sh
test:
	go test -race -test.timeout 120s $(PKG_TEST)
test-verbose:
	go test -race -test.timeout 120s -v $(PKG_TEST)
test-with-coverage:
	ginkgo -r -cover -race -skipPackage="testdata"
