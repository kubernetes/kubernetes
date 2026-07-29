#   Copyright The containerd Authors.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


# Go command to use for build
GO ?= go
INSTALL ?= install

# Root directory of the project (absolute path).
ROOTDIR=$(dir $(abspath $(lastword $(MAKEFILE_LIST))))

WHALE = "🇩"
ONI = "👹"

# Project binaries.
COMMANDS=protoc-gen-go-ttrpc

ifdef BUILDTAGS
    GO_BUILDTAGS = ${BUILDTAGS}
endif
GO_BUILDTAGS ?=
GO_TAGS=$(if $(GO_BUILDTAGS),-tags "$(strip $(GO_BUILDTAGS))",)

# Project packages.
PACKAGES=$(shell $(GO) list ${GO_TAGS} ./... | grep -v /example)
TESTPACKAGES=$(shell $(GO) list ${GO_TAGS} ./... | grep -v /cmd | grep -v /integration | grep -v /example)
BINPACKAGES=$(addprefix ./cmd/,$(COMMANDS))

#Replaces ":" (*nix), ";" (windows) with newline for easy parsing
GOPATHS=$(shell echo ${GOPATH} | tr ":" "\n" | tr ";" "\n")

TESTFLAGS_RACE=
GO_BUILD_FLAGS=
# See Golang issue re: '-trimpath': https://github.com/golang/go/issues/13809
GO_GCFLAGS=$(shell				\
	set -- ${GOPATHS};			\
	echo "-gcflags=-trimpath=$${1}/src";	\
	)

BINARIES=$(addprefix bin/,$(COMMANDS))

# Flags passed to `go test`
TESTFLAGS ?= $(TESTFLAGS_RACE) $(EXTRA_TESTFLAGS)
TESTFLAGS_PARALLEL ?= 8

# Use this to replace `go test` with, for instance, `gotestsum`
GOTEST ?= $(GO) test

.PHONY: clean all AUTHORS build binaries test integration generate protos check-protos coverage ci check help install vendor maintainer-clean
.DEFAULT: default

# Forcibly set the default goal to all, in case an include above brought in a rule definition.
.DEFAULT_GOAL := all

all: binaries

check: ## run all linters
	@echo "$(WHALE) $@"
	GOGC=75 golangci-lint run

ci: check binaries check-protos coverage # coverage-integration ## to be used by the CI

AUTHORS: .mailmap .git/HEAD
	git log --format='%aN <%aE>' | sort -fu > $@

generate: protos
	@echo "$(WHALE) $@"
	@PATH="${ROOTDIR}/bin:${PATH}" $(GO) generate -x ${PACKAGES}

protos: bin/protoc-gen-go-ttrpc ## generate protobuf
	@echo "$(WHALE) $@"
	(cd example && buf generate)
	buf generate

check-protos: protos ## check if protobufs needs to be generated again
	@echo "$(WHALE) $@"
	@test -z "$$(git status --short | grep ".pb.go" | tee /dev/stderr)" || \
		((git diff | cat) && \
		(echo "$(ONI) please run 'make protos' when making changes to proto files" && false))

build: ## build the go packages
	@echo "$(WHALE) $@"
	@$(GO) build ${DEBUG_GO_GCFLAGS} ${GO_GCFLAGS} ${GO_BUILD_FLAGS} ${EXTRA_FLAGS} ${PACKAGES}

test: ## run tests, except integration tests and tests that require root
	@echo "$(WHALE) $@"
	@$(GOTEST) ${TESTFLAGS} ${TESTPACKAGES}

integration: ## run integration tests
	@echo "$(WHALE) $@"
	@cd "${ROOTDIR}/integration" && $(GOTEST) -v ${TESTFLAGS}  -parallel ${TESTFLAGS_PARALLEL} .

benchmark: ## run benchmarks tests
	@echo "$(WHALE) $@"
	@$(GO) test ${TESTFLAGS} -bench . -run Benchmark

maintainer-clean:
	@echo "$(WHALE) $@"
	@find . -name '*.pb.go' -delete

FORCE:

define BUILD_BINARY
@echo "$(WHALE) $@"
@$(GO) build ${DEBUG_GO_GCFLAGS} ${GO_GCFLAGS} ${GO_BUILD_FLAGS} -o $@ ${GO_TAGS}  ./$<
endef

# Build a binary from a cmd.
bin/%: cmd/% FORCE
	$(call BUILD_BINARY)

binaries: $(BINARIES) ## build binaries
	@echo "$(WHALE) $@"

clean: ## clean up binaries
	@echo "$(WHALE) $@"
	@rm -f $(BINARIES)

install: ## install binaries
	@echo "$(WHALE) $@ $(BINPACKAGES)"
	@$(GO) install $(BINPACKAGES)

coverage: ## generate coverprofiles from the unit tests, except tests that require root
	@echo "$(WHALE) $@"
	@rm -f coverage.txt
	@$(GO) test ${TESTFLAGS} ${TESTPACKAGES} 2> /dev/null
	@( for pkg in ${PACKAGES}; do \
		$(GO) test ${TESTFLAGS} \
			-cover \
			-coverprofile=profile.out \
			-covermode=atomic $$pkg || exit; \
		if [ -f profile.out ]; then \
			cat profile.out >> coverage.txt; \
			rm profile.out; \
		fi; \
	done )

vendor: ## ensure all the go.mod/go.sum files are up-to-date
	@echo "$(WHALE) $@"
	@$(GO) mod tidy
	@$(GO) mod verify

verify-vendor: ## verify if all the go.mod/go.sum files are up-to-date
	@echo "$(WHALE) $@"
	@$(GO) mod tidy
	@$(GO) mod verify
	@test -z "$$(git status --short | grep "go.sum" | tee /dev/stderr)" || \
		((git diff | cat) && \
		(echo "$(ONI) make sure to checkin changes after go mod tidy" && false))

help: ## this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort
