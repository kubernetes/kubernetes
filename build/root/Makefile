# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DBG_MAKEFILE ?=
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** starting Makefile for goal(s) "$(MAKECMDGOALS)")
    $(warning ***** $(shell date))
else
    # If we're not debugging the Makefile, don't echo recipes.
    MAKEFLAGS += -s
endif


# Old-skool build tools.
#
# Commonly used targets (see each target for more information):
#   all: Build code.
#   test: Run tests.
#   clean: Clean up.

# It's necessary to set this because some environments don't link sh -> bash.
SHELL := /usr/bin/env bash -o errexit -o pipefail -o nounset
BASH_ENV := ./hack/lib/logging.sh

# Define variables so `make --warn-undefined-variables` works.
PRINT_HELP ?=
WHAT ?=
TESTS ?=

# We don't need make's built-in rules.
MAKEFLAGS += --no-builtin-rules
# Be pedantic about undefined variables.
MAKEFLAGS += --warn-undefined-variables
.SUFFIXES:

# Constants used throughout.
.EXPORT_ALL_VARIABLES:
OUT_DIR ?= _output
BIN_DIR := $(OUT_DIR)/bin

ifdef KUBE_GOFLAGS
$(info KUBE_GOFLAGS is now deprecated. Please use GOFLAGS instead.)
ifndef GOFLAGS
GOFLAGS := $(KUBE_GOFLAGS)
unexport KUBE_GOFLAGS
else
$(error Both KUBE_GOFLAGS and GOFLAGS are set. Please use just GOFLAGS)
endif
endif

# This controls the verbosity of the build.  Higher numbers mean more output.
KUBE_VERBOSE ?= 1

define ALL_HELP_INFO
# Build code.
#
# Args:
#   WHAT: Directory or Go package names to build.  If any of these directories
#   has a 'main' package, the build will produce executable files under
#   $(OUT_DIR)/bin.  If not specified, "everything" will be built.
#     "vendor/<module>/<path>" is accepted as alias for "<module>/<path>".
#     "ginkgo" is an alias for the ginkgo CLI.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags passed to 'go' when building.
#   GOGCFLAGS: Additional go compile flags passed to 'go' when building.
#   DBG: If set to "1", build with optimizations disabled for easier
#     debugging.  Any other value is ignored.
#
# Example:
#   make
#   make all
#   make all WHAT=cmd/kubelet GOFLAGS=-v
#   make all DBG=1
#     Note: Specify DBG=1 for building unstripped binaries, which allows you to
#     use code debugging tools like delve. When DBG is unspecified, it defaults
#     to "-s -w" which strips debug information.
endef
.PHONY: all
ifeq ($(PRINT_HELP),y)
all:
	echo "$$ALL_HELP_INFO"
else
all:
	hack/make-rules/build.sh $(WHAT)
endif

define GINKGO_HELP_INFO
# Build ginkgo
#
# Example:
# make ginkgo
endef
.PHONY: ginkgo
ifeq ($(PRINT_HELP),y)
ginkgo:
	echo "$$GINKGO_HELP_INFO"
else
ginkgo:
	hack/make-rules/build.sh github.com/onsi/ginkgo/v2/ginkgo
endif

define VERIFY_HELP_INFO
# Runs all the presubmission verifications.
#
# Args:
#   BRANCH: Branch to be passed to verify-vendor.sh script.
#   WHAT: List of checks to run
#
# Example:
#   make verify
#   make verify BRANCH=branch_x
#   make verify WHAT="gofmt typecheck"
endef
.PHONY: verify
ifeq ($(PRINT_HELP),y)
verify:
	echo "$$VERIFY_HELP_INFO"
else ifeq ($(origin KUBE_VERIFY_GIT_BRANCH), undefined)
verify:
	KUBE_VERIFY_GIT_BRANCH=$(BRANCH) hack/make-rules/verify.sh
else
verify:
	hack/make-rules/verify.sh
endif

define QUICK_VERIFY_HELP_INFO
# Runs only the presubmission verifications that aren't slow.
#
# Example:
#   make quick-verify
endef
.PHONY: quick-verify
ifeq ($(PRINT_HELP),y)
quick-verify:
	echo "$$QUICK_VERIFY_HELP_INFO"
else
quick-verify:
	QUICK=true SILENT=false hack/make-rules/verify.sh
endif

define UPDATE_HELP_INFO
# Runs all the update scripts.
#
# Example:
# make update
endef
.PHONY: update
ifeq ($(PRINT_HELP),y)
update:
	echo "$$UPDATE_HELP_INFO"
else
update:
	hack/make-rules/update.sh
endif

define CHECK_TEST_HELP_INFO
# Build and run tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#   TESTS: Same as WHAT.
#   KUBE_COVER: Whether to run tests with code coverage. Set to 'y' to enable coverage collection.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags to pass to 'go' when building.
#   GOGCFLAGS: Additional go compile flags passed to 'go' when building.
#
# Example:
#   make check
#   make test
#   make check WHAT=./pkg/kubelet GOFLAGS=-v
endef
.PHONY: check test
ifeq ($(PRINT_HELP),y)
check test:
	echo "$$CHECK_TEST_HELP_INFO"
else
check test:
	hack/make-rules/test.sh $(WHAT) $(TESTS)
endif

define TEST_IT_HELP_INFO
# Build and run integration tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#   KUBE_TEST_ARGS: Arguments to pass to the resulting go test invocation.
#
# Example:
#   make test-integration
#   make test-integration WHAT=./test/integration/kubelet GOFLAGS="-v -coverpkg=./pkg/kubelet/..." KUBE_COVER="y"
#   make test-integration WHAT=./test/integration/pods GOFLAGS="-v" KUBE_TEST_ARGS='-run ^TestPodUpdateActiveDeadlineSeconds$$'
endef
.PHONY: test-integration
ifeq ($(PRINT_HELP),y)
test-integration:
	echo "$$TEST_IT_HELP_INFO"
else
test-integration:
	# KUBE_TEST_ARGS is explicitly passed here in order to ensure that we are preserving any dollar signs in the value.
	KUBE_TEST_ARGS='$(value KUBE_TEST_ARGS)' hack/make-rules/test-integration.sh $(WHAT)
endif

define TEST_E2E_NODE_HELP_INFO
# Build and run node end-to-end tests.
#
# Args:
#  FOCUS: Regexp that matches the tests to be run.  Defaults to "".
#  SKIP: Regexp that matches the tests that needs to be skipped.
#    Defaults to "\[Flaky\]|\[Slow\]|\[Serial\]".
#  TEST_ARGS: A space-separated list of arguments to pass to node e2e test.
#    Defaults to "".
#  RUN_UNTIL_FAILURE: If true, pass --until-it-fails=true to ginkgo so tests are run
#    repeatedly until they fail.  Defaults to false.
#  REMOTE: If true, run the tests on a remote host.  Defaults to false.
#  REMOTE_MODE: For REMOTE=true only.  Mode for remote execution (eg. gce, ssh).
#    If set to "gce", an instance can be provisioned or reused from GCE. If set
#    to "ssh", provided `HOSTS` must be IPs or resolvable.  Defaults to "gce".
#  ARTIFACTS: Local directory to scp test artifacts into from the remote hosts
#    for REMOTE=true. Local directory to write juntil xml results into for REMOTE=false.
#    Defaults to "/tmp/_artifacts/$$(date +%y%m%dT%H%M%S)".
#  TIMEOUT: For REMOTE=true only. How long (in golang duration format) to wait
#    for ginkgo tests to complete. Defaults to 45m.
#  PARALLELISM: The number of ginkgo nodes to run.  Defaults to 8.
#  CONTAINER_RUNTIME_ENDPOINT: remote container endpoint to connect to.
#    Defaults to "/run/containerd/containerd.sock".
#  IMAGE_SERVICE_ENDPOINT: remote image endpoint to connect to, to prepull images.
#    Defaults to CONTAINER_RUNTIME_ENDPOINT.
#  IMAGE_CONFIG_FILE: path to a file containing image configuration.
#  IMAGE_CONFIG_DIR: path to image config files.
#  SYSTEM_SPEC_NAME: The name of the system spec to be used for validating the
#    image in the node conformance test. The specs are located at
#    test/e2e_node/system/specs/. For example, "SYSTEM_SPEC_NAME=gke" will use
#    the spec at test/e2e_node/system/specs/gke.yaml. If unspecified, the
#    default built-in spec (system.DefaultSpec) will be used.
#  IMAGES: For REMOTE=true only.  Comma delimited list of images for creating
#    remote hosts to run tests against.  Defaults to a recent image.
#  HOSTS: For REMOTE=true only.  Comma delimited list of running gce hosts to
#    run tests against.  Defaults to "".
#  DELETE_INSTANCES: For REMOTE=true only.  Delete any instances created as
#    part of this test run.  Defaults to false.
#  PREEMPTIBLE_INSTANCES: For REMOTE=true only.  Mark created gce instances
#    as preemptible.  Defaults to false.
#  CLEANUP: For REMOTE=true only.  If false, do not stop processes or delete
#    test files on remote hosts.  Defaults to true.
#  IMAGE_PROJECT: For REMOTE=true only.  Project containing images provided to
#    $$IMAGES.  Defaults to "cos-cloud".
#  INSTANCE_PREFIX: For REMOTE=true only.  Instances created from images will
#    have the name "$${INSTANCE_PREFIX}-$${IMAGE_NAME}".  Defaults to "test".
#  INSTANCE_METADATA: For REMOTE=true and running on GCE only.
#  GUBERNATOR: For REMOTE=true only. Produce link to Gubernator to view logs.
#    Defaults to false.
#  TEST_SUITE: For REMOTE=true only. Test suite to use. Defaults to "default".
#  SSH_USER: For REMOTE=true only  SSH username to use.
#  SSH_KEY: For REMOTE=true only. Path to SSH key to use.
#  SSH_OPTIONS: For REMOTE=true only. SSH options to use.
#  INSTANCE_TYPE: For REMOTE=true only. Machine type to use.
#  NODE_ENV: For REMOTE=true only. Additional metadata keys to add the instance.
#  RUNTIME_CONFIG: The runtime configuration for the API server on the node e2e tests.
#  E2E_TEST_DEBUG_TOOL: one of dlv/delve/gdb. Runs the test/e2e_node test binary
#    interactively under the chosen debugger. Only works for REMOTE=false and
#    in combination with DBG=1.
#
# Example:
#   make test-e2e-node FOCUS=Kubelet SKIP=container
#   make test-e2e-node REMOTE=true DELETE_INSTANCES=true
#   make test-e2e-node TEST_ARGS='--kubelet-flags="--cgroups-per-qos=true"'
# Build and run tests.
endef
.PHONY: test-e2e-node
ifeq ($(PRINT_HELP),y)
test-e2e-node:
	echo "$$TEST_E2E_NODE_HELP_INFO"
else
test-e2e-node: ginkgo
	hack/make-rules/test-e2e-node.sh
endif

define TEST_CMD_HELP_INFO
# Build and run cmdline tests.
#
# Args:
#   WHAT: List of tests to run, check test/cmd/legacy-script.sh for names.
#     For example, WHAT=deployment will run run_deployment_tests function.
# Example:
#   make test-cmd
#   make test-cmd WHAT="deployment impersonation"
endef
.PHONY: test-cmd
ifeq ($(PRINT_HELP),y)
test-cmd:
	echo "$$TEST_CMD_HELP_INFO"
else
test-cmd:
	hack/make-rules/test-cmd.sh
endif

define CLEAN_HELP_INFO
# Remove all build artifacts.
#
# Example:
#   make clean
endef
.PHONY: clean
ifeq ($(PRINT_HELP),y)
clean:
	echo "$$CLEAN_HELP_INFO"
else
clean:
	build/make-clean.sh
	hack/make-rules/clean.sh
endif

define VET_HELP_INFO
# Run 'go vet'.
#
# Args:
#   WHAT: Directory names to vet.  All *.go files under these
#     directories will be vetted.  If not specified, "everything" will be
#     vetted.
#
# Example:
#   make vet
#   make vet WHAT=./pkg/kubelet
endef
.PHONY: vet
ifeq ($(PRINT_HELP),y)
vet:
	echo "$$VET_HELP_INFO"
else
vet:
	hack/make-rules/vet.sh $(WHAT)
endif

define LINT_HELP_INFO
# Run golangci-lint
#
# Example:
#   make lint
endef
.PHONY: lint
ifeq ($(PRINT_HELP),y)
lint:
	echo "$$LINT_HELP_INFO"
else
lint:
	hack/verify-golangci-lint.sh
endif

define RELEASE_HELP_INFO
# Build a release
# Use the 'release-in-a-container' target to build the release when already in
# a container vs. creating a new container to build in using the 'release'
# target.  Useful for running in GCB.
#
# Example:
#   make release
#   make release-in-a-container
endef
.PHONY: release release-in-a-container
ifeq ($(PRINT_HELP),y)
release release-in-a-container:
	echo "$$RELEASE_HELP_INFO"
else
release release-in-a-container: KUBE_BUILD_CONFORMANCE = y
release:
	build/release.sh
release-in-a-container:
	build/release-in-a-container.sh
endif

define RELEASE_IMAGES_HELP_INFO
# Build release images
#
# Args:
#   KUBE_BUILD_CONFORMANCE: Whether to build conformance testing image as well. Set to 'n' to skip.
#   DBG: If set to "1", build with optimizations disabled for easier debugging. Any other value is ignored.
#
# Example:
#   make release-images
#   make release-images DBG=1
#     Note: Specify DBG=1 for building unstripped binaries, which allows you to use code debugging
#           tools like delve. When DBG is unspecified, it defaults to "-s -w" which strips debug
#           information.
endef
.PHONY: release-images
ifeq ($(PRINT_HELP),y)
release-images: KUBE_BUILD_CONFORMANCE = y
release-images:
	echo "$$RELEASE_IMAGES_HELP_INFO"
else
release-images:
	build/release-images.sh
endif

define RELEASE_SKIP_TESTS_HELP_INFO
# Build a release, but skip tests
#
# Args:
#   KUBE_RELEASE_RUN_TESTS: Whether to run tests. Set to 'y' to run tests anyways.
#   KUBE_FASTBUILD: Whether to cross-compile for other architectures. Set to 'false' to do so.
#   KUBE_DOCKER_REGISTRY: Registry of released images, default to registry.k8s.io
#   KUBE_BASE_IMAGE_REGISTRY: Registry of base images for controlplane binaries, default to registry.k8s.io/build-image
#
# Example:
#   make release-skip-tests
#   make quick-release
endef
.PHONY: release-skip-tests quick-release
ifeq ($(PRINT_HELP),y)
release-skip-tests quick-release:
	echo "$$RELEASE_SKIP_TESTS_HELP_INFO"
else
release-skip-tests quick-release: KUBE_RELEASE_RUN_TESTS = n
release-skip-tests quick-release: KUBE_FASTBUILD = true
release-skip-tests quick-release:
	build/release.sh
endif

define QUICK_RELEASE_IMAGES_HELP_INFO
# Build release images, but only for linux/amd64
#
# Args:
#   KUBE_FASTBUILD: Whether to cross-compile for other architectures. Set to 'false' to do so.
#   KUBE_BUILD_CONFORMANCE: Whether to build conformance testing image as well. Set to 'y' to do so.
#   DBG: If set to "1", build with optimizations disabled for easier debugging. Any other value is ignored.
#
# Example:
#   make quick-release-images
#   make quick-release-images DBG=1
#     Note: Specify DBG=1 for building unstripped binaries, which allows you to use code debugging
#           tools like delve. When DBG is unspecified, it defaults to "-s -w" which strips debug
#           information.
endef
.PHONY: quick-release-images
ifeq ($(PRINT_HELP),y)
quick-release-images:
	echo "$$QUICK_RELEASE_IMAGES_HELP_INFO"
else
quick-release-images: KUBE_FASTBUILD = true
quick-release-images:
	build/release-images.sh
endif

define PACKAGE_HELP_INFO
# Package tarballs
# Use the 'package-tarballs' target to run the final packaging steps of
# a release.
#
# Example:
#   make package-tarballs
endef
.PHONY: package package-tarballs
ifeq ($(PRINT_HELP),y)
package package-tarballs:
	echo "$$PACKAGE_HELP_INFO"
else
package-tarballs: KUBE_BUILD_CONFORMANCE = y
package package-tarballs:
	build/package-tarballs.sh
endif

define CROSS_HELP_INFO
# Cross-compile for all platforms
# Use the 'cross-in-a-container' target to cross build when already in
# a container vs. creating a new container to build from (build-image)
# Useful for running in GCB.
#
# Example:
#   make cross
#   make cross-in-a-container
endef
.PHONY: cross cross-in-a-container
ifeq ($(PRINT_HELP),y)
cross cross-in-a-container:
	echo "$$CROSS_HELP_INFO"
else
cross cross-in-a-container: KUBE_BUILD_CONFORMANCE = y
cross:
	hack/make-rules/cross.sh
cross-in-a-container: KUBE_OUTPUT_SUBPATH = $(OUT_DIR)/dockerized
cross-in-a-container:
ifeq (,$(wildcard /.dockerenv))
	echo -e "\nThe 'cross-in-a-container' target can only be used from within a docker container.\n"
else
	hack/make-rules/cross.sh
endif
endif

define CMD_HELP_INFO
# Add rules for all directories in cmd/
#
# Example:
#   make kubectl kube-proxy
endef
#TODO: make EXCLUDE_TARGET auto-generated when there are other files in cmd/
EXCLUDE_TARGET=OWNERS
CMD_TARGET = $(filter-out %$(EXCLUDE_TARGET),$(notdir $(abspath $(wildcard cmd/*/))))
.PHONY: $(CMD_TARGET)
ifeq ($(PRINT_HELP),y)
$(CMD_TARGET):
	echo "$$CMD_HELP_INFO"
else
$(CMD_TARGET):
	hack/make-rules/build.sh cmd/$@
endif

define HELP_INFO
# Print make targets and help info
#
# Example:
# make help
endef
.PHONY: help
ifeq ($(PRINT_HELP),y)
help:
	echo "$$HELP_INFO"
else
help:
	hack/make-rules/make-help.sh
endif
