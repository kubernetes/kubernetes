# Simple targets:
#
# 'make' : build all binaries and store them in OUT_DIR/go/bin
# 'make kubelet' : build the kubelet binary (see BINS)
# 'make kubelet' GOFLAGS=-v : build the kubelet binary and print progress
# 'make WHAT=pkg/kubelet' : run 'go build' in an arbitrary dir
#
# 'make test' : build and run all tests
# 'make test WHAT=pkg/kubelet' : build and run tests in pkg/kubelet
#
# 'make clean' : remove all build artifacts

SHELL = bash
GO = go
GODEP = godep

CORE_BINS = cmd/proxy \
            cmd/apiserver \
            cmd/controller-manager \
            cmd/kubelet \
            cmd/kubecfg \
            cmd/integration
PLUG_BINS = plugin/cmd/scheduler
BINS = $(CORE_BINS) $(PLUG_BINS)

K8S_GO_PKG = github.com/GoogleCloudPlatform/kubernetes
OUT_DIR = output
GO_OUT_DIR = $(OUT_DIR)/go/
GO_SRC_DIR = $(GO_OUT_DIR)/src
GO_BIN_DIR = $(GO_OUT_DIR)/bin
K8S_SRC_DIR = $(GO_SRC_DIR)/$(K8S_GO_PKG)

# Standard args for 'find' to filter out noise.
FIND_FILTER = -not \( \( \
	            -wholename ./$(OUT_DIR)/\* \
	            -o -wholename \*/third_party/\* \
	            -o -wholename \*/Godeps/\* \
	          \) -prune \)


all: checkdeps $(if $(WHAT), build_some, build_all)

clean:
	@rm -rf $(OUT_DIR)

checkdeps: check_go check_godep check_go_ver

check_go:
	@set -e -u; \
	if [[ -z "$$(which $(GO))" ]]; then \
	  echo "Can't find 'go' in PATH, please fix and retry." >&2; \
	  echo "See http://golang.org/doc/install for installation instructions." >&2; \
	  false; \
	fi

check_godep:
	@set -e -u; \
	if [[ -z "$$(which $(GODEP))" ]]; then \
	  echo "Can't find 'godep' in PATH, please fix and retry." >&2; \
	  echo "See https://github.com/GoogleCloudPlatform/kubernetes#godep-and-dependency-management" >&2; \
	  false; \
	fi

# Travis continuous build uses a head go release that doesn't report a version
# number, so we skip this check on Travis.  It's unnecessary there anyway.
check_go_ver:
	@set -e -u; \
	if [[ "$${TRAVIS:-}" != "true" ]]; then \
	  GO_VERSION=($$($(GO) version)); \
	  if [[ "$${GO_VERSION[2]}" < "go1.2" ]]; then \
	    echo "Detected go version: $${GO_VERSION[*]}." >&2; \
	    echo "Kubernetes requires go version 1.2 or greater." >&2; \
	    false; \
	  fi; \
	fi

# This is establishing a fake Go root dir in the form that go wants, but that
# our code does not necessarily conform to.  See the expansion of K8S_SRC_DIR.
$(K8S_SRC_DIR):
	@mkdir -p $$(dirname $@)
	@ln -snf ../../../../.. $@

VERFILE = .version

# TODO: When we start making tags, switch to git describe?
.PHONY: $(VERFILE)
$(VERFILE):
	@set -e -u; \
	git_commit=$$(git rev-parse --short "HEAD^{commit}"); \
	dirty=$$(git status --porcelain); \
	echo $$git_commit$$([[ -z "$$dirty" ]] || echo -dirty) > $@.tmp; \
	if ! diff $@.tmp $@ >/dev/null 2>&1; then \
	  cat $@.tmp > $@; \
	fi; \
	rm -f $@.tmp;

clean: clean_verfile
clean_verfile:
	@rm -f $(VERFILE)

# The flag to 'go build' and 'go install' that sets version info.
VERFLAG = -ldflags "-X $(K8S_GO_PKG)/pkg/version.commitFromGit $$(cat $(VERFILE))"

build_all: $(BINS)

# Allow arbitrary 'go build' in a directory.  $(WHAT) is the target dir.
build_some: $(K8S_SRC_DIR) $(VERFILE)
	@set -e -u; \
	GOPATH=$$(pwd)/$(OUT_DIR)/go; \
	for tgt in $(WHAT); do \
	  $(GODEP) go build $(GOFLAGS) $(VERFLAG) $(K8S_GO_PKG)/$$tgt; \
	done

# All the defined binaries use the same recipe.
#
# Note that the flags to 'go install' are duplicated in the salt build setup for
# our cluster deploy.  If we add more command line options to our standard build
# we'll want to duplicate them there.  As we move to distributing pre-built
# binaries we can eliminate this duplication.
$(BINS): $(K8S_SRC_DIR) $(VERFILE)
	@GOPATH=$$(pwd)/$(OUT_DIR)/go; \
	$(GODEP) go install $(GOFLAGS) $(VERFLAG) $(K8S_GO_PKG)/$@

# Allow 'make foo' to alias to 'make cmd/foo'
$(patsubst cmd/%,%,$(CORE_BINS)): % : cmd/%
$(patsubst plugin/cmd/%,%,$(PLUG_BINS)): % : plugin/cmd/%

# Flags for 'make test'.  Any of these can be overridden on the commandline.
TEST_RACE = -race
TEST_COVER = -cover -covermode atomic
TEST_TIMEOUT = -timeout 30s
COVERAGE_DIR = /tmp/k8s_coverage

# If WHAT is set to a specific directory (e.g. pkg/kubecfg) this will only run
# tests for that dir.  If it is not set, this will run all tests.
test: $(if $(WHAT), test_some, test_all)

# Tests individual rules with coverage profile (slow).
test_some:
	@set -e -u; \
	GOPATH=$$(pwd)/$(OUT_DIR)/go; \
	for tgt in $(WHAT); do \
	  mkdir -p $(COVERAGE_DIR)/$$tgt; \
	  PROF="-coverprofile=$(COVERAGE_DIR)/$$tgt/coverage.out"; \
	  $(GODEP) go test \
	      $(TEST_RACE) \
	      $(TEST_COVER) \
	      $(TEST_TIMEOUT) $$PROF \
	      $(GOFLAGS) $(K8S_GO_PKG)/$$tgt; \
	done;

# Tests all rules, no coverage profile.
test_all:
	@set -e -u; \
	GOPATH=$$(pwd)/$(OUT_DIR)/go; \
	dirs=$$(find -name '*_test.go' $(FIND_FILTER) -print0 \
	    | xargs -0 -n1 dirname | sort -u \
	    | xargs -n1 printf "$(K8S_GO_PKG)/%s\n"); \
	$(GODEP) go test \
	    $(TEST_RACE) \
	    $(TEST_COVER) \
	    $(TEST_TIMEOUT) \
	    $(GOFLAGS) $$dirs;

# TODO: add hack/test-cmd and integration-test as tests under a new rule,
#       discovered by some naming convention?

gofmt: check_go_ver
	@find -type f -name '*.go' $(FIND_FILTER) -print0 | xargs -0 gofmt -s -w

check_gofmt: check_go_ver
	@set -e -u; \
	bad=$$(find -type f -name '*.go' $(FIND_FILTER) -print0 \
	    | xargs -0 gofmt -s -l); \
	echo $$bad | sed 's/ /\n/'; \
	if [[ -n "$$bad" ]]; then \
	  false; \
	fi

# TODO: support boilerplate verification
