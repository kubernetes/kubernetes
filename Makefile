# Old-skool build tools.
#
# Targets (see each target for more information):
#   all: Build code.
#   check: Run tests.
#   test: Run tests.
#   clean: Clean up.

OUT_DIR = _output
GODEPS_PKG_DIR = Godeps/_workspace/pkg

KUBE_GOFLAGS = $(GOFLAGS)
export KUBE_GOFLAGS

KUBE_GOLDFLAGS = $(GOLDFLAGS)
export KUBE_GOLDFLAGS

# Build code.
#
# Args:
#   WHAT: Directory names to build.  If any of these directories has a 'main'
#     package, the build will produce executable files under $(OUT_DIR)/go/bin.
#     If not specified, "everything" will be built.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make
#   make all
#   make all WHAT=cmd/kubelet GOFLAGS=-v
all:
	hack/build-go.sh $(WHAT)
.PHONY: all

# Runs all the presubmission verifications.
#
# Args:
#   BRANCH: Branch to be passed to hack/verify-godeps.sh script.
#
# Example:
#   make verify
#   make verify BRANCH=branch_x
verify:
	hack/verify-gofmt.sh
	hack/verify-boilerplate.sh
	hack/verify-codecgen.sh
	hack/verify-description.sh
	hack/verify-generated-conversions.sh
	hack/verify-generated-deep-copies.sh
	hack/verify-generated-docs.sh
	hack/verify-swagger-spec.sh
	hack/verify-linkcheck.sh
	hack/verify-flags-underscore.py
	hack/verify-godeps.sh $(BRANCH)
.PHONY: verify

# Build and run tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#   TESTS: Same as WHAT.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make check
#   make test
#   make check WHAT=pkg/kubelet GOFLAGS=-v
check test:
	hack/test-go.sh $(WHAT) $(TESTS)
.PHONY: check test

# Build and run integration tests.
#
# Example:
#   make test_integration
test_integration:
	hack/test-integration.sh
.PHONY: test_integration test_integ

# Build and run end-to-end tests.
#
# Example:
#   make test_e2e
test_e2e:
	hack/e2e-test.sh
.PHONY: test_e2e

# Remove all build artifacts.
#
# Example:
#   make clean
clean:
	build/make-clean.sh
	rm -rf $(OUT_DIR)
	rm -rf $(GODEPS_PKG_DIR)
.PHONY: clean

# Run 'go vet'.
#
# Args:
#   WHAT: Directory names to vet.  All *.go files under these
#     directories will be vetted.  If not specified, "everything" will be
#     vetted.
#   TESTS: Same as WHAT.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#
# Example:
#   make vet
#   make vet WHAT=pkg/kubelet
vet:
	hack/vet-go.sh $(WHAT) $(TESTS)
.PHONY: vet

# Build a release
#
# Example:
#   make release
release:
	build/release.sh
.PHONY: release

# Build a release, but skip tests
#
# Example:
#   make release-skip-tests
release-skip-tests quick-release:
	KUBE_RELEASE_RUN_TESTS=n build/release.sh
.PHONY: release-skip-tests quick-release

