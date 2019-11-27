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
SHELL := /bin/bash

# We don't need make's built-in rules.
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

# Constants used throughout.
.EXPORT_ALL_VARIABLES:
OUT_DIR ?= _output
BIN_DIR := $(OUT_DIR)/bin
PRJ_SRC_PATH := k8s.io/kubernetes
GENERATED_FILE_PREFIX := zz_generated.

# Metadata for driving the build lives here.
META_DIR := .make

ifdef KUBE_GOFLAGS
$(info KUBE_GOFLAGS is now deprecated. Please use GOFLAGS instead.)
ifndef GOFLAGS
GOFLAGS := $(KUBE_GOFLAGS)
unexport KUBE_GOFLAGS
else
$(error Both KUBE_GOFLAGS and GOFLAGS are set. Please use just GOFLAGS)
endif
endif

# Extra options for the release or quick-release options:
KUBE_RELEASE_RUN_TESTS := $(KUBE_RELEASE_RUN_TESTS)
KUBE_FASTBUILD := $(KUBE_FASTBUILD)

# This controls the verbosity of the build.  Higher numbers mean more output.
KUBE_VERBOSE ?= 1

define ALL_HELP_INFO
# Build code.
#
# Args:
#   WHAT: Directory names to build.  If any of these directories has a 'main'
#     package, the build will produce executable files under $(OUT_DIR)/go/bin.
#     If not specified, "everything" will be built.
#   GOFLAGS: Extra flags to pass to 'go' when building.
#   GOLDFLAGS: Extra linking flags passed to 'go' when building.
#   GOGCFLAGS: Additional go compile flags passed to 'go' when building.
#
# Example:
#   make
#   make all
#   make all WHAT=cmd/kubelet GOFLAGS=-v
#   make all GOLDFLAGS=""
#     Note: Specify GOLDFLAGS as an empty string for building unstripped binaries, which allows
#           you to use code debugging tools like delve. When GOLDFLAGS is unspecified, it defaults
#           to "-s -w" which strips debug information. Other flags that can be used for GOLDFLAGS 
#           are documented at https://golang.org/cmd/link/
endef
.PHONY: all
ifeq ($(PRINT_HELP),y)
all:
	@echo "$$ALL_HELP_INFO"
else
all: generated_files
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
	@echo "$$GINKGO_HELP_INFO"
else
ginkgo:
	hack/make-rules/build.sh vendor/github.com/onsi/ginkgo/ginkgo
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
#   make verify WHAT="bazel typecheck"
endef
.PHONY: verify
ifeq ($(PRINT_HELP),y)
verify:
	@echo "$$VERIFY_HELP_INFO"
else
verify:
	KUBE_VERIFY_GIT_BRANCH=$(BRANCH) hack/make-rules/verify.sh
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
	@echo "$$QUICK_VERIFY_HELP_INFO"
else
quick-verify:
	QUICK=true SILENT=false hack/make-rules/verify.sh
endif

define UPDATE_HELP_INFO
# Runs all the generated updates.
#
# Example:
# make update
endef
.PHONY: update
ifeq ($(PRINT_HELP),y)
update:
	@echo "$$UPDATE_HELP_INFO"
else
update: generated_files
	CALLED_FROM_MAIN_MAKEFILE=1 hack/make-rules/update.sh
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
	@echo "$$CHECK_TEST_HELP_INFO"
else
check test: generated_files
	hack/make-rules/test.sh $(WHAT) $(TESTS)
endif

define TEST_IT_HELP_INFO
# Build and run integration tests.
#
# Args:
#   WHAT: Directory names to test.  All *_test.go files under these
#     directories will be run.  If not specified, "everything" will be tested.
#
# Example:
#   make test-integration
endef
.PHONY: test-integration
ifeq ($(PRINT_HELP),y)
test-integration:
	@echo "$$TEST_IT_HELP_INFO"
else
test-integration: generated_files
	hack/make-rules/test-integration.sh $(WHAT)
endif

define TEST_E2E_NODE_HELP_INFO
# Build and run node end-to-end tests.
#
# Args:
#  FOCUS: Regexp that matches the tests to be run.  Defaults to "".
#  SKIP: Regexp that matches the tests that needs to be skipped.  Defaults
#    to "".
#  RUN_UNTIL_FAILURE: If true, pass --untilItFails to ginkgo so tests are run
#    repeatedly until they fail.  Defaults to false.
#  REMOTE: If true, run the tests on a remote host instance on GCE.  Defaults
#    to false.
#  IMAGES: For REMOTE=true only.  Comma delimited list of images for creating
#    remote hosts to run tests against.  Defaults to a recent image.
#  LIST_IMAGES: If true, don't run tests.  Just output the list of available
#    images for testing.  Defaults to false.
#  HOSTS: For REMOTE=true only.  Comma delimited list of running gce hosts to
#    run tests against.  Defaults to "".
#  DELETE_INSTANCES: For REMOTE=true only.  Delete any instances created as
#    part of this test run.  Defaults to false.
#  PREEMPTIBLE_INSTANCES: For REMOTE=true only.  Mark created gce instances
#    as preemptible.  Defaults to false.
#  ARTIFACTS: For REMOTE=true only.  Local directory to scp test artifacts into
#    from the remote hosts.  Defaults to "/tmp/_artifacts".
#  REPORT: For REMOTE=false only.  Local directory to write juntil xml results
#    to.  Defaults to "/tmp/".
#  CLEANUP: For REMOTE=true only.  If false, do not stop processes or delete
#    test files on remote hosts.  Defaults to true.
#  IMAGE_PROJECT: For REMOTE=true only.  Project containing images provided to
#  IMAGES.  Defaults to "kubernetes-node-e2e-images".
#  INSTANCE_PREFIX: For REMOTE=true only.  Instances created from images will
#    have the name "${INSTANCE_PREFIX}-${IMAGE_NAME}".  Defaults to "test".
#  INSTANCE_METADATA: For REMOTE=true and running on GCE only.
#  GUBERNATOR: For REMOTE=true only. Produce link to Gubernator to view logs.
#	 Defaults to false.
#  PARALLELISM: The number of gingko nodes to run.  Defaults to 8.
#  RUNTIME: Container runtime to use (eg. docker, remote).
#    Defaults to "docker".
#  CONTAINER_RUNTIME_ENDPOINT: remote container endpoint to connect to.
#   Used when RUNTIME is set to "remote".
#  IMAGE_SERVICE_ENDPOINT: remote image endpoint to connect to, to prepull images.
#   Used when RUNTIME is set to "remote".
#  IMAGE_CONFIG_FILE: path to a file containing image configuration.
#  SYSTEM_SPEC_NAME: The name of the system spec to be used for validating the
#    image in the node conformance test. The specs are located at
#    test/e2e_node/system/specs/. For example, "SYSTEM_SPEC_NAME=gke" will use
#    the spec at test/e2e_node/system/specs/gke.yaml. If unspecified, the
#    default built-in spec (system.DefaultSpec) will be used.
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
	@echo "$$TEST_E2E_NODE_HELP_INFO"
else
test-e2e-node: ginkgo generated_files
	hack/make-rules/test-e2e-node.sh
endif

define TEST_E2E_KUBEADM_HELP_INFO
# Build and run kubeadm end-to-end tests.
#
# Args:
#  FOCUS: Regexp that matches the tests to be run.  Defaults to "".
#  SKIP: Regexp that matches the tests that needs to be skipped.  Defaults
#    to "".
#  RUN_UNTIL_FAILURE: If true, pass --untilItFails to ginkgo so tests are run
#    repeatedly until they fail. Defaults to false.
#  ARTIFACTS: Local directory to save test artifacts into. Defaults to "/tmp/_artifacts".
#  PARALLELISM: The number of gingko nodes to run.  If empty ginkgo default 
#    parallelism (cores - 1) is used
#  BUILD: Build kubeadm end-to-end tests. Defaults to true.
#
# Example:
#   make test-e2e-kubeadm 
#   make test-e2e-kubeadm FOCUS=kubeadm-config 
#   make test-e2e-kubeadm SKIP=kubeadm-config
#
# Build and run tests.
endef
.PHONY: test-e2e-kubeadm
ifeq ($(PRINT_HELP),y)
test-e2e-kubeadm:
	@echo "$$TEST_E2E_KUBEADM_HELP_INFO"
else
test-e2e-kubeadm: 
	hack/make-rules/test-e2e-kubeadm.sh
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
	@echo "$$TEST_CMD_HELP_INFO"
else
test-cmd: generated_files
	hack/make-rules/test-cmd.sh
endif

define CLEAN_HELP_INFO
# Remove all build artifacts.
#
# Example:
#   make clean
#
# TODO(thockin): call clean_generated when we stop committing generated code.
endef
.PHONY: clean
ifeq ($(PRINT_HELP),y)
clean:
	@echo "$$CLEAN_HELP_INFO"
else
clean: clean_meta
	build/make-clean.sh
	hack/make-rules/clean.sh
endif

define CLEAN_META_HELP_INFO
# Remove make-related metadata files.
#
# Example:
#   make clean_meta
endef
.PHONY: clean_meta
ifeq ($(PRINT_HELP),y)
clean_meta:
	@echo "$$CLEAN_META_HELP_INFO"
else
clean_meta:
	rm -rf $(META_DIR)
endif

define CLEAN_GENERATED_HELP_INFO
# Remove all auto-generated artifacts. Generated artifacts in staging folder should not be removed as they are not
# generated using generated_files.
#
# Example:
#   make clean_generated
endef
.PHONY: clean_generated
ifeq ($(PRINT_HELP),y)
clean_generated:
	@echo "$$CLEAN_GENERATED_HELP_INFO"
else
clean_generated:
	find . -type f -name $(GENERATED_FILE_PREFIX)\* | grep -v "[.]/staging/.*" | xargs rm -f
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
	@echo "$$VET_HELP_INFO"
else
vet: generated_files
	CALLED_FROM_MAIN_MAKEFILE=1 hack/make-rules/vet.sh $(WHAT)
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
	@echo "$$RELEASE_HELP_INFO"
else
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
#
# Example:
#   make release-images
endef
.PHONY: release-images
ifeq ($(PRINT_HELP),y)
release-images:
	@echo "$$RELEASE_IMAGES_HELP_INFO"
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
#   KUBE_DOCKER_REGISTRY: Registry of released images, default to k8s.gcr.io
#   KUBE_BASE_IMAGE_REGISTRY: Registry of base images for controlplane binaries, default to k8s.gcr.io
#
# Example:
#   make release-skip-tests
#   make quick-release
endef
.PHONY: release-skip-tests quick-release
ifeq ($(PRINT_HELP),y)
release-skip-tests quick-release:
	@echo "$$RELEASE_SKIP_TESTS_HELP_INFO"
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
#   KUBE_BUILD_CONFORMANCE: Whether to build conformance testing image as well. Set to 'n' to skip.
#
# Example:
#   make quick-release-images
endef
.PHONY: quick-release-images
ifeq ($(PRINT_HELP),y)
quick-release-images:
	@echo "$$QUICK_RELEASE_IMAGES_HELP_INFO"
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
	@echo "$$PACKAGE_HELP_INFO"
else
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
	@echo "$$CROSS_HELP_INFO"
else
cross:
	hack/make-rules/cross.sh
cross-in-a-container: KUBE_OUTPUT_SUBPATH = $(OUT_DIR)/dockerized
cross-in-a-container:
ifeq (,$(wildcard /.dockerenv))
	@echo -e "\nThe 'cross-in-a-container' target can only be used from within a docker container.\n"
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
EXCLUDE_TARGET=BUILD OWNERS
.PHONY: $(filter-out %$(EXCLUDE_TARGET),$(notdir $(abspath $(wildcard cmd/*/))))
ifeq ($(PRINT_HELP),y)
$(filter-out %$(EXCLUDE_TARGET),$(notdir $(abspath $(wildcard cmd/*/)))):
	@echo "$$CMD_HELP_INFO"
else
$(filter-out %$(EXCLUDE_TARGET),$(notdir $(abspath $(wildcard cmd/*/)))): generated_files
	hack/make-rules/build.sh cmd/$@
endif

define GENERATED_FILES_HELP_INFO
# Produce auto-generated files needed for the build.
#
# Example:
#   make generated_files
endef
.PHONY: generated_files
ifeq ($(PRINT_HELP),y)
generated_files:
	@echo "$$GENERATED_FILES_HELP_INFO"
else
generated_files gen_openapi:
	$(MAKE) -f Makefile.generated_files $@ CALLED_FROM_MAIN_MAKEFILE=1
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
	@echo "$$HELP_INFO"
else
help:
	hack/make-rules/make-help.sh
endif

# Non-dockerized bazel rules.
.PHONY: bazel-build bazel-test bazel-release

ifeq ($(PRINT_HELP),y)
define BAZEL_BUILD_HELP_INFO
# Build with bazel
#
# Example:
# make bazel-build
endef
bazel-build:
	@echo "$$BAZEL_BUILD_HELP_INFO"
else
# Some things in vendor don't build due to empty target lists for cross-platform rules.
bazel-build:
	bazel build -- //... -//vendor/...
endif


ifeq ($(PRINT_HELP),y)
define BAZEL_TEST_HELP_INFO
# Test with bazel
#
# Example:
# make bazel-test
endef
bazel-test:
	@echo "$$BAZEL_TEST_HELP_INFO"
else
# //hack:verify-all is a manual target.
# Some things in vendor don't build due to empty target lists for cross-platform rules.
bazel-test:
	bazel test --config=unit -- \
	  //... \
	  //hack:verify-all \
	  -//vendor/...
endif

ifeq ($(PRINT_HELP),y)
define BAZEL_TEST_INTEGRATION_HELP_INFO
# Integration test with bazel
#
# Example:
# make bazel-test-integration
endef
bazel-test-integration:
	@echo "$$BAZEL_TEST_INTEGRATION_HELP_INFO"
else
bazel-test-integration:
	bazel test --config integration //test/integration/...
endif

ifeq ($(PRINT_HELP),y)
define BAZEL_RELEASE_HELP_INFO
# Build release tars with bazel
#
# Example:
# make bazel-release
endef
bazel-release:
	@echo "$$BAZEL_RELEASE_HELP_INFO"
else
bazel-release:
	bazel build //build/release-tars
endif
