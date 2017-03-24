# Root directory of the project (absolute path).
ROOTDIR=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Base path used to install.
DESTDIR=/usr/local

# Used to populate version variable in main package.
VERSION=$(shell git describe --match 'v[0-9]*' --dirty='.m' --always)

PKG=github.com/docker/containerd

# Project packages.
PACKAGES=$(shell go list ./... | grep -v /vendor/)
INTEGRATION_PACKAGE=${PKG}/integration
SNAPSHOT_PACKAGES=$(shell go list ./snapshot/...)

# Project binaries.
COMMANDS=ctr containerd containerd-shim protoc-gen-gogoctrd dist ctrd-protobuild
BINARIES=$(addprefix bin/,$(COMMANDS))

GO_LDFLAGS=-ldflags "-X $(PKG).Version=$(VERSION) -X $(PKG).Package=$(PKG)"

# Flags passed to `go test`
TESTFLAGS ?=-parallel 8 -race

.PHONY: clean all AUTHORS fmt vet lint build binaries test integration setup generate protos checkprotos coverage ci check help install uninstall vendor
.DEFAULT: default

all: binaries

check: fmt vet lint ineffassign ## run fmt, vet, lint, ineffassign

ci: check binaries checkprotos coverage coverage-integration ## to be used by the CI

AUTHORS: .mailmap .git/HEAD
	git log --format='%aN <%aE>' | sort -fu > $@

setup: ## install dependencies
	@echo "🐳 $@"
	# TODO(stevvooe): Install these from the vendor directory
	@go get -u github.com/golang/lint/golint
	#@go get -u github.com/kisielk/errcheck
	@go get -u github.com/gordonklaus/ineffassign

generate: protos
	@echo "🐳 $@"
	@PATH=${ROOTDIR}/bin:${PATH} go generate -x ${PACKAGES}

protos: bin/protoc-gen-gogoctrd bin/ctrd-protobuild ## generate protobuf
	@echo "🐳 $@"
	@PATH=${ROOTDIR}/bin:${PATH} ctrd-protobuild ${PACKAGES}

checkprotos: protos ## check if protobufs needs to be generated again
	@echo "🐳 $@"
	@test -z "$$(git status --short | grep ".pb.go" | tee /dev/stderr)" || \
		((git diff | cat) && \
		(echo "👹 please run 'make generate' when making changes to proto files" && false))

# Depends on binaries because vet will silently fail if it can't load compiled
# imports
vet: binaries ## run go vet
	@echo "🐳 $@"
	@test -z "$$(go vet ${PACKAGES} 2>&1 | grep -v 'constant [0-9]* not a string in call to Errorf' | egrep -v '(timestamp_test.go|duration_test.go|exit status 1)' | tee /dev/stderr)"

fmt: ## run go fmt
	@echo "🐳 $@"
	@test -z "$$(gofmt -s -l . | grep -v vendor/ | grep -v ".pb.go$$" | tee /dev/stderr)" || \
		(echo "👹 please format Go code with 'gofmt -s -w'" && false)
	@test -z "$$(find . -path ./vendor -prune -o ! -name timestamp.proto ! -name duration.proto -name '*.proto' -type f -exec grep -Hn -e "^ " {} \; | tee /dev/stderr)" || \
		(echo "👹 please indent proto files with tabs only" && false)
	@test -z "$$(find . -path ./vendor -prune -o -name '*.proto' -type f -exec grep -EHn "[_ ]id = " {} \; | grep -v gogoproto.customname | tee /dev/stderr)" || \
		(echo "👹 id fields in proto files must have a gogoproto.customname set" && false)
	@test -z "$$(find . -path ./vendor -prune -o -name '*.proto' -type f -exec grep -Hn "Meta meta = " {} \; | grep -v '(gogoproto.nullable) = false' | tee /dev/stderr)" || \
		(echo "👹 meta fields in proto files must have option (gogoproto.nullable) = false" && false)

lint: ## run go lint
	@echo "🐳 $@"
	@test -z "$$(golint ./... | grep -v vendor/ | grep -v ".pb.go:" | tee /dev/stderr)"

ineffassign: ## run ineffassign
	@echo "🐳 $@"
	@test -z "$$(ineffassign . | grep -v vendor/ | grep -v ".pb.go:" | tee /dev/stderr)"

#errcheck: ## run go errcheck
#	@echo "🐳 $@"
#	@test -z "$$(errcheck ./... | grep -v vendor/ | grep -v ".pb.go:" | tee /dev/stderr)"

build: ## build the go packages
	@echo "🐳 $@"
	@go build -i -v ${GO_LDFLAGS} ${GO_GCFLAGS} ${PACKAGES}

test: ## run tests, except integration tests and tests that require root
	@echo "🐳 $@"
	@go test ${TESTFLAGS} $(filter-out ${INTEGRATION_PACKAGE},${PACKAGES})

root-test: ## run tests, except integration tests
	@echo "🐳 $@"
	@go test ${TESTFLAGS} ${SNAPSHOT_PACKAGES} -test.root

integration: ## run integration tests
	@echo "🐳 $@"
	@go test ${TESTFLAGS} ${INTEGRATION_PACKAGE}

FORCE:

# Build a binary from a cmd.
bin/%: cmd/% FORCE
	@test $$(go list) = "${PKG}" || \
		(echo "👹 Please correctly set up your Go build environment. This project must be located at <GOPATH>/src/${PKG}" && false)
	@echo "🐳 $@"
	@go build -i -o $@ ${GO_LDFLAGS}  ${GO_GCFLAGS} ./$<

binaries: $(BINARIES) ## build binaries
	@echo "🐳 $@"

clean: ## clean up binaries
	@echo "🐳 $@"
	@rm -f $(BINARIES)

install: ## install binaries
	@echo "🐳 $@ $(BINARIES)"
	@mkdir -p $(DESTDIR)/bin
	@install $(BINARIES) $(DESTDIR)/bin

uninstall:
	@echo "🐳 $@"
	@rm -f $(addprefix $(DESTDIR)/bin/,$(notdir $(BINARIES)))


coverage: ## generate coverprofiles from the unit tests, except tests that require root
	@echo "🐳 $@"
	@rm -f coverage.txt
	( for pkg in $(filter-out ${INTEGRATION_PACKAGE},${PACKAGES}); do \
		go test -i ${TESTFLAGS} -test.short -coverprofile=coverage.out -covermode=atomic $$pkg || exit; \
		if [ -f profile.out ]; then \
			cat profile.out >> coverage.txt; \
			rm profile.out; \
		fi; \
		go test ${TESTFLAGS} -test.short -coverprofile=coverage.out -covermode=atomic $$pkg || exit; \
		if [ -f profile.out ]; then \
			cat profile.out >> coverage.txt; \
			rm profile.out; \
		fi; \
	done )

root-coverage: ## generae coverage profiles for the unit tests
	@echo "🐳 $@"
	@( for pkg in ${SNAPSHOT_PACKAGES}; do \
		go test -i ${TESTFLAGS} -test.short -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg -test.root || exit; \
		go test ${TESTFLAGS} -test.short -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg -test.root || exit; \
	done )

coverage-integration: ## generate coverprofiles from the integration tests
	@echo "🐳 $@"
	go test ${TESTFLAGS} -test.short -coverprofile="../../../${INTEGRATION_PACKAGE}/coverage.txt" -covermode=atomic ${INTEGRATION_PACKAGE}

vendor:
	@echo "🐳 $@"
	@vndr

help: ## this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort
