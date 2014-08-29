# Old-skool build tools.
#
# Targets (see each target for more information):
#   all: Build code.
#   check: Run tests.
#   test: Run tests.
#   clean: Clean up.

OUT_DIR = _output

export GOFLAGS

# Build code.
#
# Args:
#   WHAT: Directory names to build.  If any of these directories has a 'main'
#     package, the build will produce executable files under $(OUT_DIR)/go/bin.
#     If not specified, "everything" will be built.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#
# Example:
#   make
#   make all
#   make all WHAT=cmd/kubelet GOFLAGS=-v
all:
	hack/build-go.sh $(WHAT)
.PHONY: all

# Build and run tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#   TESTS: Same as WHAT.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#
# Example:
#   make check
#   make test
#   make check WHAT=pkg/kubelet GOFLAGS=-v
check test:
	hack/test-go.sh $(WHAT) $(TESTS)
.PHONY: check test

# Remove all build artifacts.
#
# Example:
#   make clean
clean:
	rm -rf $(OUT_DIR)
.PHONY: clean
