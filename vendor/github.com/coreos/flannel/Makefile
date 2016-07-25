.PHONY: all test cover gofmt gofmt-fix license-check

# Grab the absolute directory that contains this file.
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# These variables can be overridden by setting an environment variable.
TEST_PACKAGES?=pkg/ip subnet remote
TEST_PACKAGES_EXPANDED=$(TEST_PACKAGES:%=github.com/coreos/flannel/%)
PACKAGES?=$(TEST_PACKAGES) network
PACKAGES_EXPANDED=$(PACKAGES:%=github.com/coreos/flannel/%)

default: help
all: test				    ## Run all the tests
binary: artifacts/flanneld  ## Create the flanneld binary

artifacts/flanneld: $(shell find . -type f  -name '*.go')
	mkdir -p artifacts
	go build -o artifacts/flanneld \
	  -ldflags "-extldflags -static -X github.com/coreos/flannel/version.Version=$(shell git describe --dirty)"

test:
	go test -cover $(TEST_PACKAGES_EXPANDED)

cover:
	#A single package must be given - e.g. 'PACKAGES=pkg/ip make cover'
	go test -coverprofile cover.out $(PACKAGES_EXPANDED)
	go tool cover -html=cover.out


# Throw an error if gofmt finds problems.
# "read" will return a failure return code if there is no output. This is inverted wth the "!"
gofmt:
	! gofmt -d $(PACKAGES) 2>&1 | read

gofmt-fix:
	gofmt -w $(PACKAGES)

license-check:
	dist/license-check.sh

## Display this help text
help: # Some kind of magic from https://gist.github.com/rcmachado/af3db315e31383502660
	$(info Available targets)
	@awk '/^[a-zA-Z\-\_0-9]+:/ {								   \
		nb = sub( /^## /, "", helpMsg );							 \
		if(nb == 0) {												\
			helpMsg = $$0;											 \
			nb = sub( /^[^:]*:.* ## /, "", helpMsg );				  \
		}															\
		if (nb)													  \
			printf "\033[1;31m%-" width "s\033[0m %s\n", $$1, helpMsg; \
	}															  \
	{ helpMsg = $$0 }'											 \
	$(MAKEFILE_LIST)
